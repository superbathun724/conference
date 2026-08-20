"""모든 물리 채널이 구현해야 하는 공통 인터페이스.

새 채널(음향, 화면, 확장 채널)을 추가할 때 이 인터페이스만 채우면
core/의 프레임·파운틴·측정 코드를 그대로 재사용할 수 있다.
자세한 설계 이유는 docs/ARCHITECTURE.md 참고.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelCaps:
    """채널의 물리적 성질. 비교표를 자동 생성하는 데 쓴다."""

    name: str  # "acoustic_fsk"
    medium: str  # "air" | "solid" | "em_visible" | "em_rf" | "magnetic"
    wave_type: str  # "longitudinal" | "electromagnetic" | "quasi_static"
    coding_axes: tuple  # ("time",) 또는 ("time", "space")
    directional: bool  # 지향성 여부
    penetrates_opaque: bool  # 불투명 차폐물 통과 여부
    human_perceptible: bool  # 사람이 신호의 존재를 알아챌 수 있는가
    nominal_bps: float  # 설계상 목표 전송률 (실측치 아님)


class Channel(ABC):
    """비트 배열 ↔ 물리 신호 변환과, 실제 장치 입출력을 분리한 추상 클래스.

    modulate/demodulate는 장치 없이 순수 함수로 검증할 수 있어야 하고,
    emit/capture만 실제 하드웨어를 건드린다. 이 분리 덕분에 스피커나
    카메라 없이도 루프백(loopback 함수)으로 로직을 개발할 수 있다.
    """

    caps: ChannelCaps

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """비트 배열(0/1) → 물리 신호 표현.
        음향이면 파형 샘플, 화면이면 프레임 이미지 배열."""

    @abstractmethod
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """물리 신호 표현 → 비트 배열. 복조 실패 시 빈 배열."""

    @abstractmethod
    def emit(self, signal: np.ndarray) -> None:
        """실제 장치로 내보낸다 (스피커 재생, 화면 표시)."""

    @abstractmethod
    def capture(self, duration_s: float) -> np.ndarray:
        """실제 장치에서 받는다 (마이크 녹음, 카메라 캡처)."""


def loopback(channel: Channel, bits: np.ndarray, snr_db: float | None = None) -> np.ndarray:
    """modulate → (선택적 잡음) → demodulate 를 장치 없이 직접 연결한다.

    잡음 없는 루프백에서 비트가 하나라도 틀리면 물리 문제가 아니라
    코드 버그다. 이 구분이 디버깅 시간을 가장 많이 줄인다.
    """
    signal = channel.modulate(bits)
    if snr_db is not None:
        signal = add_awgn(signal, snr_db)
    return channel.demodulate(signal)


def add_awgn(signal: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """신호에 지정한 SNR(dB)만큼 백색 가우시안 잡음을 더한다.

    난수는 반드시 시드를 받는다 — 같은 시드면 같은 잡음이 나와야
    실험을 재현할 수 있다.
    """
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear if snr_linear > 0 else 0.0
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise
