"""FSK(주파수 편이 변조, Frequency-Shift Keying)로 비트를 소리에 싣는 음향 채널.

비트 0과 1을 서로 다른 두 반송 주파수의 톤(tone)으로 표현한다. 수신 측은
심볼(비트 하나를 나타내는 고정 길이 구간)마다 FFT를 걸어 두 반송 주파수
성분 중 어느 쪽이 더 강한지 보고 비트를 판정한다.

심볼 길이와 주파수 분해능은 서로 맞바꿈 관계다. FFT의 주파수 분해능 Δf는
대략 1 / (심볼 길이)로 정해진다. 심볼을 짧게 해서 전송 속도를 올리면 Δf가
커져 두 반송 주파수를 구별하기 어려워지고, 두 주파수 간격보다 Δf가 커지는
순간 오류율이 급격히 오른다. 이 맞바꿈이 docs/ARCHITECTURE.md가 말하는
축 A·축 B 공통 제약(Δt·Δf ≳ 상수)의 실물이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml

from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.core.channel import Channel, ChannelCaps


@dataclass(frozen=True)
class AcousticFskConfig:
    """음향 FSK의 물리 파라미터. 실험 시에는 config/channels/acoustic_fsk.yaml에서 읽는다."""

    sample_rate_hz: int = 44100
    freq0_hz: float = 3000.0  # 비트 0을 나타내는 반송 주파수
    freq1_hz: float = 4000.0  # 비트 1을 나타내는 반송 주파수
    symbol_duration_ms: float = 20.0
    amplitude: float = 0.5
    edge_ramp_ms: float = 2.0  # 심볼 경계 클릭 잡음(스펙트럼 누설) 방지용 램프 길이
    preamble_detect_threshold: float = 0.3  # 정규화 상관값 임계치 (0~1)
    # 심볼 판정에 쓰는 가운데 구간의 비율. 심볼 경계는 앞 심볼의 잔향과 램프가
    # 섞이는 자리라, 가운데만 보면 잔향(심볼 간 간섭)을 덜 탄다. 1.0이면 전체 사용
    guard_fraction: float = 0.6
    # 프리앰블 탐색 걸음(ms). 심볼 길이보다 훨씬 짧아야 시작점을 정확히 잡는다
    preamble_search_step_ms: float = 1.0
    # 심볼 주기 추정 범위(±%)와 걸음(%). 재생 기기와 녹음 기기의 시계가 0.2%만 어긋나도
    # 300비트 프레임 끝에서 심볼이 반 칸 이상 밀려 CRC가 깨진다. 0이면 추정하지 않는다
    clock_search_percent: float = 1.5
    clock_search_step_percent: float = 0.01
    input_device: int | None = None
    output_device: int | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> AcousticFskConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class AcousticFsk(Channel):
    """두 반송 주파수로 0/1을 표현하는 음향 FSK 채널."""

    def __init__(self, config: AcousticFskConfig | None = None) -> None:
        self.config = config or AcousticFskConfig()
        symbols_per_sec = 1000.0 / self.config.symbol_duration_ms
        self.caps = ChannelCaps(
            name="acoustic_fsk",
            medium="air",
            wave_type="longitudinal",
            coding_axes=("time",),
            directional=False,
            penetrates_opaque=False,
            human_perceptible=True,
            nominal_bps=symbols_per_sec,  # 심볼 하나당 비트 하나
        )
        self._symbol_len = round(self.config.symbol_duration_ms / 1000 * self.config.sample_rate_hz)
        ramp_len = round(self.config.edge_ramp_ms / 1000 * self.config.sample_rate_hz)
        self._ramp_len = min(ramp_len, self._symbol_len // 2)

    # ---- 변조/복조: 장치 없이 순수 계산만 한다 (루프백 가능) ----

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """비트 배열 → 심볼 파형을 이어붙인 신호."""
        if len(bits) == 0:
            return np.array([], dtype=np.float64)
        symbols = [self._make_symbol(int(bit)) for bit in bits]
        return np.concatenate(symbols)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """신호 → 비트 배열. 프리앰블을 못 찾으면 빈 배열을 돌려준다."""
        start = self._find_preamble_start(signal)
        if start is None:
            return np.array([], dtype=np.uint8)

        # 시작점이 한두 표본 늦게 잡히면 마지막 심볼이 창에 못 들어가 비트 하나가
        # 빠진다. 끝에 심볼 하나만큼 0을 붙여 두면 마지막 심볼은 언제나 들어가고,
        # 그 뒤에 생기는 잡음 비트는 프레임의 길이 필드가 잘라낸다.
        signal = np.concatenate([signal, np.zeros(self._symbol_len)])

        period = self._estimate_symbol_period(signal, start)
        soft = self._soft_bits_at(signal, start, period)
        return (soft > 0).astype(np.uint8)

    def _soft_bits_at(self, signal: np.ndarray, start: int, period: float) -> np.ndarray:
        """start부터 period 간격으로 심볼을 잘라 각 심볼의 (E1−E0)/(E1+E0)를 돌려준다.

        period는 정수가 아니어도 된다. 심볼 i의 시작을 start + i·period로 두고
        가장 가까운 표본으로 반올림하므로, 시계 오차로 생기는 미세한 밀림이
        누적되지 않는다.
        """
        sym = self._symbol_len
        core = self._core(np.arange(sym))
        t = core / self.config.sample_rate_hz
        k0 = np.exp(-2j * np.pi * self.config.freq0_hz * t)
        k1 = np.exp(-2j * np.pi * self.config.freq1_hz * t)
        n_sym = int((len(signal) - start - sym) // period) + 1
        if n_sym <= 0:
            return np.array([], dtype=np.float64)
        starts = start + np.round(np.arange(n_sym) * period).astype(int)
        idx = starts[:, None] + core[None, :]
        seg = signal[idx]
        e0 = np.abs(seg @ k0)
        e1 = np.abs(seg @ k1)
        return (e1 - e0) / np.maximum(e1 + e0, 1e-12)

    def _estimate_symbol_period(self, signal: np.ndarray, start: int) -> float:
        """이 녹음에서 실제 심볼 주기(표본 수)를 추정한다.

        왜 필요한가. 송신 기기의 재생 시계와 수신 기기의 녹음 시계는 같지 않다.
        폰 재생기가 44.1kHz 파일을 48kHz로 바꾸어 트는 과정만으로도 0.1~0.5%의
        차이가 생길 수 있는데, 2026-09-05 1m 측정에서 프리앰블은 잡히고 CRC만
        깨진 양상은 시뮬레이션에서 0.2% 오차로 정확히 재현됐다. 16비트 프리앰블은
        밀림이 작아 살아남고, 그 뒤 300비트를 지나며 누적된 밀림이 심볼 경계를
        넘겨 버리는 것이다.

        어떻게 하는가. 설정 주기를 ±clock_search_percent 범위에서 조금씩 바꿔
        가며 복조해 보고, 판정이 가장 또렷한 주기를 고른다. 또렷함의 척도는
        |E1−E0|/(E1+E0)의 평균이다. 주기가 맞으면 매 창이 심볼 하나에 정확히
        얹혀 비율이 ±1에 가깝고, 주기가 틀리면 창이 두 심볼에 걸쳐 비율이 0
        쪽으로 뭉개진다. 신호가 끝난 뒤의 무음 구간은 에너지 문턱으로 걸러
        척도에서 뺀다. 이 판단은 채널 안에서 신호만 보고 하며 프레임의 CRC를
        들여다보지 않는다 — 계층을 넘지 않기 위해서다.
        """
        sym = float(self._symbol_len)
        pct = float(self.config.clock_search_percent)
        step = float(self.config.clock_search_step_percent)
        if pct <= 0 or step <= 0:
            return sym

        core = self._core(np.arange(self._symbol_len))
        t = core / self.config.sample_rate_hz
        k0 = np.exp(-2j * np.pi * self.config.freq0_hz * t)
        k1 = np.exp(-2j * np.pi * self.config.freq1_hz * t)

        # 신호 유무 문턱: 프리앰블 16심볼의 에너지 중앙값의 20%
        pre = signal[start : start + self._symbol_len * 16]
        n_pre = len(pre) // self._symbol_len
        if n_pre == 0:
            return sym
        block = pre[: n_pre * self._symbol_len].reshape(n_pre, self._symbol_len)[:, core]
        gate = 0.2 * float(np.median(np.abs(block @ k0) + np.abs(block @ k1)))

        best_period, best_score = sym, -np.inf
        for scale in np.arange(-pct, pct + 1e-9, step) / 100.0:
            period = sym * (1.0 + scale)
            n_sym = int((len(signal) - start - self._symbol_len) // period) + 1
            if n_sym < 16:
                continue
            starts = start + np.round(np.arange(n_sym) * period).astype(int)
            seg = signal[starts[:, None] + core[None, :]]
            e0 = np.abs(seg @ k0)
            e1 = np.abs(seg @ k1)
            total = e0 + e1
            active = total > gate
            if active.sum() < 16:
                continue
            score = float(np.mean(np.abs(e1 - e0)[active] / total[active]))
            if score > best_score:
                best_period, best_score = period, score
        return best_period

    # ---- 실제 장치 ----

    def emit(self, signal: np.ndarray) -> None:
        """스피커로 재생한다."""
        sd.play(signal, samplerate=self.config.sample_rate_hz, device=self.config.output_device)
        sd.wait()

    def capture(self, duration_s: float) -> np.ndarray:
        """마이크로 녹음한다."""
        n_samples = round(duration_s * self.config.sample_rate_hz)
        recording = sd.rec(
            n_samples,
            samplerate=self.config.sample_rate_hz,
            channels=1,
            device=self.config.input_device,
        )
        sd.wait()
        return recording[:, 0]

    # ---- 내부 구현 ----

    def _make_symbol(self, bit: int) -> np.ndarray:
        """비트 하나 → 톤 파형 하나.

        심볼 경계에서 진폭이 뚝 끊기면 스피커에서 클릭음이 나고, 그 클릭음이
        넓은 주파수 대역에 잡음을 뿌려 다음 심볼의 FFT 판정을 흐린다. 경계에
        반코사인 램프를 씌워 진폭을 부드럽게 올리고 내린다.
        """
        freq_hz = self.config.freq1_hz if bit else self.config.freq0_hz
        t = np.arange(self._symbol_len) / self.config.sample_rate_hz
        wave = self.config.amplitude * np.sin(2 * np.pi * freq_hz * t)

        if self._ramp_len > 0:
            ramp = (1 - np.cos(np.linspace(0, np.pi, self._ramp_len))) / 2
            wave[: self._ramp_len] *= ramp
            wave[-self._ramp_len :] *= ramp[::-1]
        return wave

    def _decode_symbol(self, window: np.ndarray) -> int:
        """심볼 구간의 두 반송 주파수 성분 크기를 비교해 비트를 정한다.

        심볼 전체가 아니라 가운데 guard_fraction만큼만 본다. 심볼 경계에는 앞
        심볼의 잔향(벽 반사음이 늦게 도착한 것)과 램프가 겹쳐 있어, 거기까지
        포함하면 이웃 심볼의 톤이 섞여 든다. 2026-09-05 1m 측정에서 CRC 불일치가
        늘어난 원인으로 잔향을 의심했고, 이 처리는 그에 대한 대응이다.
        """
        e0, e1 = self._tone_energies(self._core(window))
        return 1 if e1 > e0 else 0

    def _core(self, window: np.ndarray) -> np.ndarray:
        """심볼 창의 가운데 guard_fraction 구간만 잘라낸다."""
        frac = float(np.clip(self.config.guard_fraction, 0.1, 1.0))
        n = len(window)
        keep = max(8, int(round(n * frac)))
        start = (n - keep) // 2
        return window[start : start + keep]

    def _tone_energies(self, window: np.ndarray) -> tuple[float, float]:
        """두 반송 주파수 각각에 대한 성분 크기.

        FFT의 격자에 맞추는 대신 정확히 그 주파수의 복소 지수에 투영한다
        (Goertzel과 같은 계산). 창 길이가 무엇이든 3,000 Hz와 4,000 Hz를 정확히
        본다. FFT 격자에 맞추면 창을 짧게 잘랐을 때 반송 주파수가 격자 사이에
        떨어져 이웃 격자로 새는 문제가 있다.
        """
        t = np.arange(len(window)) / self.config.sample_rate_hz
        e0 = abs(np.sum(window * np.exp(-2j * np.pi * self.config.freq0_hz * t)))
        e1 = abs(np.sum(window * np.exp(-2j * np.pi * self.config.freq1_hz * t)))
        return float(e0), float(e1)

    def _find_preamble_start(self, signal: np.ndarray) -> int | None:
        """프리앰블의 시작 표본 위치를 찾는다 — 파형이 아니라 **톤 에너지 비율**로.

        2026-09-05 이전에는 프리앰블 파형과 수신 신호를 그대로 상관시켰다. 그
        방식은 두 가지에 취약했다. 첫째, 상관값을 송신 파형의 에너지로만 나눴기
        때문에 수신 진폭이 작으면(8월 측정에서 0.24~0.92배로 4배 변동) 임계값에
        못 미쳐 "프리앰블 미검출"이 났다. 둘째, 파형 상관은 위상에 민감해서
        스피커-공기-마이크를 거치며 위상이 틀어지거나 잔향이 섞이면 상관 봉우리가
        뭉개진다.

        지금 방식은 위상과 진폭을 아예 보지 않는다. 신호를 1ms 걸음으로 훑으며
        각 위치에서 심볼 길이만큼의 창을 잡아 두 톤의 에너지 비율
        s = (E1 − E0) / (E1 + E0) ∈ [−1, 1] 을 구한다. 이것은 "이 자리가 1에
        가까운가 0에 가까운가"를 뜻하는 부드러운 비트값이다. 프리앰블 16비트를
        ±1로 바꿔 같은 시간축에 펼친 뒤 이 부드러운 비트열과 상관시키면, 패턴이
        맞는 자리에서만 봉우리가 선다. 비율은 진폭과 무관하고, 에너지는 위상과
        무관하다.

        정규화 상관값은 완전히 맞을 때 1이며 임계값(preamble_detect_threshold)과
        비교한다.
        """
        preamble_bits = bytes_to_bits(frame.PREAMBLE)
        n_pre = len(preamble_bits)
        step = max(
            1, round(self.config.preamble_search_step_ms / 1000 * self.config.sample_rate_hz)
        )
        sym = self._symbol_len
        if len(signal) < sym * n_pre:
            return None

        # 가운데 구간만 보는 투영 벡터. 창 길이가 달라도 정확히 그 주파수를 본다.
        core = self._core(np.arange(sym))
        t = core / self.config.sample_rate_hz
        k0 = np.exp(-2j * np.pi * self.config.freq0_hz * t)
        k1 = np.exp(-2j * np.pi * self.config.freq1_hz * t)

        def soft_bits(x: np.ndarray, stride: int) -> np.ndarray:
            """x를 stride 걸음으로 훑으며 각 위치의 창(길이 sym)에 대해 (E1−E0)/(E1+E0)."""
            windows = np.lib.stride_tricks.sliding_window_view(x, sym)[::stride]
            seg = windows[:, core]
            e0 = np.abs(seg @ k0)
            e1 = np.abs(seg @ k1)
            return (e1 - e0) / np.maximum(e1 + e0, 1e-12)

        # 1) 거친 탐색. 창은 그 위치에서 **시작**하지만, 패턴의 비트 i는 심볼 i의
        #    **가운데**와 비교해야 맞는다. 그래서 앞에 반 심볼만큼 0을 붙여 두면
        #    패딩된 좌표의 창 시작 = 원래 좌표의 심볼 가운데가 되어 좌표가 맞아떨어진다.
        half = sym // 2
        padded = np.concatenate([np.zeros(half), signal])
        soft = soft_bits(padded, step)

        per_symbol = max(1, round(sym / step))
        pattern = np.repeat(np.where(preamble_bits, 1.0, -1.0), per_symbol)
        if len(soft) < len(pattern):
            return None
        corr = np.correlate(soft, pattern, mode="valid") / len(pattern)
        coarse = int(np.argmax(corr))
        if corr[coarse] < self.config.preamble_detect_threshold:
            return None
        start = coarse * step  # 패딩 덕분에 이 값이 곧 원래 좌표의 프레임 시작

        # 2) 미세 탐색. 거친 걸음 안에서 표본 단위로 가장 잘 맞는 자리를 다시 찾는다.
        best_start, best_score = start, -np.inf
        lo = max(0, start - step)
        hi = min(len(signal) - sym * n_pre, start + step)
        ideal = np.where(preamble_bits, 1.0, -1.0)
        for cand in range(lo, hi + 1):
            block = signal[cand : cand + sym * n_pre].reshape(n_pre, sym)[:, core]
            e0 = np.abs(block @ k0)
            e1 = np.abs(block @ k1)
            score = float(np.mean(((e1 - e0) / np.maximum(e1 + e0, 1e-12)) * ideal))
            if score > best_score:
                best_start, best_score = cand, score
        return best_start
