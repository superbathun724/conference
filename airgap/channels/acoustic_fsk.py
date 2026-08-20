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

        bits = []
        pos = start
        while pos + self._symbol_len <= len(signal):
            window = signal[pos : pos + self._symbol_len]
            bits.append(self._decode_symbol(window))
            pos += self._symbol_len
        return np.array(bits, dtype=np.uint8)

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
        """심볼 구간의 FFT에서 두 반송 주파수 성분 크기를 비교해 비트를 정한다."""
        spectrum = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), d=1 / self.config.sample_rate_hz)
        bin0 = int(np.argmin(np.abs(freqs - self.config.freq0_hz)))
        bin1 = int(np.argmin(np.abs(freqs - self.config.freq1_hz)))
        return 1 if abs(spectrum[bin1]) > abs(spectrum[bin0]) else 0

    def _find_preamble_start(self, signal: np.ndarray) -> int | None:
        """프리앰블 파형과 수신 신호를 상관시켜 프레임 시작 표본 위치를 찾는다.

        바커 부호 계열 프리앰블(core/frame.py)은 자기 자신과 겹칠 때만 상관값이
        크고 다른 위치·다른 비트열과는 상관값이 작게 나오도록 설계돼 있다.
        전체 신호를 훑으며 프리앰블 파형과의 상관을 계산해 가장 큰 지점을 찾고,
        그 값이 "완전히 일치했을 때의 에너지" 대비 임계치를 못 넘으면 프레임이
        없다고 본다.
        """
        preamble_bits = bytes_to_bits(frame.PREAMBLE)
        preamble_wave = self.modulate(preamble_bits)
        if len(signal) < len(preamble_wave):
            return None

        correlation = np.correlate(signal, preamble_wave, mode="valid")
        peak_index = int(np.argmax(np.abs(correlation)))
        reference_energy = float(np.sum(preamble_wave**2))
        if reference_energy == 0:
            return None
        normalized_peak = abs(correlation[peak_index]) / reference_energy
        if normalized_peak < self.config.preamble_detect_threshold:
            return None
        return peak_index
