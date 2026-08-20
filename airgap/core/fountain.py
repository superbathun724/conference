"""LT(Luby Transform) 파운틴 부호.

원본을 K개 조각으로 나눈 뒤, 그중 무작위 개수(차수)를 XOR한 새 조각("방울",
droplet)을 시드 기반으로 끝없이 만들어낸다. 수신측은 어떤 방울을 어떤 순서로
받든 K보다 살짝 많은 개수만 모이면 원본을 복원할 수 있다 — 되돌아갈 통신로가
없어 재전송을 요청할 수 없는 오프라인 채널에 이 성질이 꼭 맞는다. 방울마다
"어떤 원본 조각을 섞었는지" 목록을 통째로 보내는 대신, 그 방울을 만들 때 쓴
난수 시드 하나만 보내면 수신측이 같은 절차로 같은 조각들을 다시 뽑아낼 수 있어
전송 오버헤드가 작다 (core/frame.py의 16비트 시드 필드가 이 값을 나른다).

차수를 고르는 확률표(로버스트 솔리톤 분포)가 이 부호의 핵심 설계다. 차수 1인
방울(원본 그대로)이 항상 어느 정도 나오게 하고 나머지는 낮은 차수에 몰아,
디코더가 "차수 1 확정 → 다른 방울에서 XOR로 소거 → 새 차수 1 발생" 연쇄
(펼치기, peeling)를 끊기지 않고 이어갈 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class LtConfig:
    """LT 부호 파라미터. config/fountain.yaml에서 읽는다."""

    chunk_size_bytes: int = 16
    c: float = 0.1
    delta: float = 0.05

    @classmethod
    def from_yaml(cls, path: Path) -> LtConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass(frozen=True)
class Droplet:
    """방울 하나. seed만 있으면 payload를 만들 때 섞은 원본 인덱스를 다시 구할 수 있다."""

    seed: int
    degree: int
    payload: bytes


def split_into_chunks(data: bytes, chunk_size_bytes: int) -> list[bytes]:
    """바이트열을 고정 길이 조각 K개로 나눈다. 마지막 조각은 0으로 채운다."""
    if chunk_size_bytes <= 0:
        raise ValueError(f"조각 크기는 1바이트 이상이어야 한다: {chunk_size_bytes}")

    chunks = [data[i : i + chunk_size_bytes] for i in range(0, len(data), chunk_size_bytes)]
    if not chunks:
        chunks = [b""]

    padding = chunk_size_bytes - len(chunks[-1])
    if padding > 0:
        chunks[-1] = chunks[-1] + bytes(padding)
    return chunks


def assemble_chunks(chunks: dict[int, bytes], original_length: int) -> bytes:
    """복원된 조각들을 원래 순서로 이어붙이고, 마지막 조각에 넣었던 0 패딩을 잘라낸다."""
    ordered = b"".join(chunks[i] for i in range(len(chunks)))
    return ordered[:original_length]


def robust_soliton_distribution(k: int, c: float, delta: float) -> np.ndarray:
    """차수 1..k를 고를 확률표(로버스트 솔리톤 분포, Luby 2002)를 만든다.

    이상적 솔리톤 분포 ρ만 쓰면 초기에 차수 1인 방울이 너무 드물게 나와
    디코더가 펼치기를 시작하지 못하는 경우가 생긴다. 여기에 특정 차수
    부근(대략 k/S)에 확률을 몰아주는 τ를 더해, 차수 1이 꾸준히 공급되도록
    보정한 것이 로버스트 솔리톤 분포다. c와 delta는 그 보정의 폭과, 목표로
    하는 복호 실패 확률 상한을 정한다.
    """
    if k <= 0:
        raise ValueError(f"k는 1 이상이어야 한다: {k}")

    degrees = np.arange(1, k + 1, dtype=np.float64)

    ideal = np.empty(k)
    ideal[0] = 1.0 / k
    if k > 1:
        ideal[1:] = 1.0 / (degrees[1:] * (degrees[1:] - 1))

    spike_width = max(c * np.sqrt(k) * np.log(k / delta), 1.0)  # 논문 표기로는 S
    boundary = min(max(round(k / spike_width), 1), k)  # 스파이크가 몰리는 차수, K/S를 반올림

    correction = np.zeros(k)
    if boundary > 1:
        correction[: boundary - 1] = spike_width / (degrees[: boundary - 1] * k)
    correction[boundary - 1] += spike_width * np.log(spike_width / delta) / k

    weights = ideal + correction
    return weights / weights.sum()


def _degree_and_indices(seed: int, k: int, distribution: np.ndarray) -> tuple[int, list[int]]:
    """시드 하나로 (차수, 이번 방울에 섞을 원본 인덱스들)을 정한다.

    송신측과 수신측이 완전히 같은 절차(같은 시드·같은 k·같은 분포)를 거치므로,
    수신측은 시드만 보고 방울에 어떤 원본이 섞였는지 스스로 재현할 수 있다.
    """
    rng = np.random.default_rng(seed)
    degree = int(rng.choice(np.arange(1, k + 1), p=distribution))
    indices = rng.choice(k, size=degree, replace=False)
    return degree, sorted(int(i) for i in indices)


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """같은 길이의 두 바이트열을 비트 단위로 XOR한다."""
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(len(a), "big")


def encode_droplet(chunks: list[bytes], seed: int, config: LtConfig) -> Droplet:
    """원본 조각들과 시드 하나로 방울 하나를 만든다."""
    k = len(chunks)
    distribution = robust_soliton_distribution(k, config.c, config.delta)
    degree, indices = _degree_and_indices(seed, k, distribution)

    payload = chunks[indices[0]]
    for idx in indices[1:]:
        payload = _xor_bytes(payload, chunks[idx])
    return Droplet(seed=seed, degree=degree, payload=payload)


class LtDecoder:
    """받은 방울들을 모아 원본 K개 조각을 복원하는 펼치기(peeling) 디코더.

    방울을 받는 족족 반영한다. 차수 1(원본 그대로인) 방울이 확정되면, 그
    인덱스를 아직 안 풀린 다른 방울들에서 XOR로 지운다 — 그러다 어떤 방울이
    차수 1이 되면 다시 확정하는 연쇄가 이어진다. 이 연쇄가 K개를 다 풀 때까지
    이어지면 복원 성공이고, 중간에 멈추면(차수 2 이상만 남고 더 못 줄어들면)
    지금까지 받은 방울로는 복원할 수 없다는 뜻이다.
    """

    def __init__(self, k: int, config: LtConfig) -> None:
        self.k = k
        self.config = config
        self._distribution = robust_soliton_distribution(k, config.c, config.delta)
        self._decoded: dict[int, bytes] = {}
        self._pending: list[tuple[set[int], bytes]] = []
        self.droplets_received = 0

    @property
    def is_complete(self) -> bool:
        return len(self._decoded) == self.k

    def add_droplet(self, seed: int, payload: bytes) -> None:
        self.droplets_received += 1
        if self.is_complete:
            return

        _, indices = _degree_and_indices(seed, self.k, self._distribution)
        self._pending.append((set(indices), payload))
        self._peel()

    def _peel(self) -> None:
        """확정된 조각들을 대기 중인 방울에서 지우고, 그러다 차수 1이 되면 확정하기를 반복한다."""
        progress = True
        while progress:
            progress = False
            still_pending = []
            for indices, payload in self._pending:
                known = indices & self._decoded.keys()
                if known:
                    for idx in known:
                        payload = _xor_bytes(payload, self._decoded[idx])
                    indices = indices - known

                if not indices:
                    continue  # 이미 다 아는 조각들의 조합이었다 — 새 정보 없음
                if len(indices) == 1:
                    (idx,) = tuple(indices)
                    self._decoded[idx] = payload
                    progress = True
                else:
                    still_pending.append((indices, payload))
            self._pending = still_pending

    def assemble(self, original_length: int) -> bytes:
        if not self.is_complete:
            raise RuntimeError(f"아직 {self.k - len(self._decoded)}개 조각이 부족해 복원할 수 없다")
        return assemble_chunks(self._decoded, original_length)
