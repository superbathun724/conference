"""밝기 변조(2준위 밝기 편이 변조)로 비트를 화면 밝기에 싣는 가시광 채널.

화면 전체를 단색으로 채워, 어두운 준위와 밝은 준위 두 가지로 비트 0/1을
표현한다. 음향 FSK가 "어느 주파수가 강한가"로 비트를 정하듯, 이 채널은
"심볼 구간 동안 카메라가 잡은 평균 밝기가 어느 준위에 가까운가"로 비트를
정한다 — 판정 원리는 같고 실어 나르는 물리량(주파수 대 밝기)만 다르다.

공간 구조 없이 시간축 하나만 쓴다는 점이 QR 채널(screen_qr.py)과의 핵심
차이다. 심볼 길이를 줄이면 전송 속도는 오르지만, 카메라가 한 심볼 구간
안에서 평균 낼 수 있는 프레임 수가 줄어 밝기 판정이 흔들린다 — 음향에서
심볼을 줄이면 주파수 분해능이 나빠지는 것과 같은 형태의 시간·분해능
맞바꿈이다(docs/ARCHITECTURE.md 참고).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.core.channel import Channel, ChannelCaps

_DISPLAY_HEIGHT_PX = 480
_DISPLAY_WIDTH_PX = 640


@dataclass(frozen=True)
class ScreenFlickerConfig:
    """밝기 변조 채널의 파라미터. config/channels/screen_flicker.yaml에서 읽는다."""

    fps: float = 30.0  # 화면 갱신률이자 카메라 표본 목표 주기
    symbol_duration_ms: float = 100.0  # 심볼 하나(비트 하나)의 길이
    level0: float = 0.05  # 비트 0 밝기 (0=완전히 어두움, 1=완전히 밝음)
    level1: float = 0.95  # 비트 1 밝기
    preamble_detect_threshold: float = 0.3  # 정규화 상관값 임계치 (0~1)
    camera_index: int | None = None
    fullscreen: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> ScreenFlickerConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ScreenFlicker(Channel):
    """화면 밝기 두 준위로 0/1을 표현하는 가시광 채널."""

    def __init__(self, config: ScreenFlickerConfig | None = None) -> None:
        self.config = config or ScreenFlickerConfig()
        symbols_per_sec = 1000.0 / self.config.symbol_duration_ms
        self.caps = ChannelCaps(
            name="screen_flicker",
            medium="em_visible",
            wave_type="electromagnetic",
            coding_axes=("time",),
            directional=True,
            penetrates_opaque=False,
            human_perceptible=True,
            nominal_bps=symbols_per_sec,
        )
        self._frames_per_symbol = max(
            1, round(self.config.fps * self.config.symbol_duration_ms / 1000)
        )
        self._midpoint = (self.config.level0 + self.config.level1) / 2

    # ---- 변조/복조: 장치 없이 밝기값 배열만 다룬다 (루프백 가능) ----

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """비트 배열 → 프레임(화면 갱신) 단위 밝기값을 이어붙인 신호."""
        if len(bits) == 0:
            return np.array([], dtype=np.float64)
        levels = np.where(bits, self.config.level1, self.config.level0)
        return np.repeat(levels, self._frames_per_symbol).astype(np.float64)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """신호 → 비트 배열. 프리앰블을 못 찾으면 빈 배열을 돌려준다."""
        start = self._find_preamble_start(signal)
        if start is None:
            return np.array([], dtype=np.uint8)

        bits = []
        pos = start
        while pos + self._frames_per_symbol <= len(signal):
            window = signal[pos : pos + self._frames_per_symbol]
            bits.append(self._decode_symbol(window))
            pos += self._frames_per_symbol
        return np.array(bits, dtype=np.uint8)

    # ---- 실제 장치 ----

    def emit(self, signal: np.ndarray) -> None:
        """화면 전체를 신호값 순서대로 단색으로 채워 표시한다."""
        window = "airgap_screen_flicker"
        flag = cv2.WINDOW_NORMAL if not self.config.fullscreen else cv2.WND_PROP_FULLSCREEN
        cv2.namedWindow(window, flag)
        if self.config.fullscreen:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        frame_delay_ms = max(1, round(1000 / self.config.fps))
        for level in signal:
            gray_value = int(round(np.clip(level, 0.0, 1.0) * 255))
            image = np.full((_DISPLAY_HEIGHT_PX, _DISPLAY_WIDTH_PX), gray_value, dtype=np.uint8)
            cv2.imshow(window, image)
            cv2.waitKey(frame_delay_ms)
        cv2.destroyWindow(window)

    def capture(self, duration_s: float) -> np.ndarray:
        """카메라 프레임의 평균 밝기를 fps 목표 주기로 표본화한다."""
        index = self.config.camera_index if self.config.camera_index is not None else 0
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없다 (camera_index={index})")

        samples = []
        try:
            frame_interval_s = 1.0 / self.config.fps
            deadline = time.monotonic() + duration_s
            next_sample_at = time.monotonic()
            while time.monotonic() < deadline:
                ok, cam_frame = cap.read()
                if not ok:
                    continue
                now = time.monotonic()
                if now >= next_sample_at:
                    gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
                    samples.append(float(gray.mean()) / 255.0)
                    next_sample_at += frame_interval_s
        finally:
            cap.release()
        return np.array(samples, dtype=np.float64)

    # ---- 내부 구현 ----

    def _decode_symbol(self, window: np.ndarray) -> int:
        """심볼 구간의 평균 밝기가 두 준위의 중간값보다 높으면 1로 판정한다."""
        return 1 if float(np.mean(window)) >= self._midpoint else 0

    def _find_preamble_start(self, signal: np.ndarray) -> int | None:
        """프리앰블 밝기 패턴과 수신 신호를 상관시켜 프레임 시작 지점을 찾는다.

        원리는 음향 채널(acoustic_fsk.py)의 프리앰블 검출과 동일하다 —
        신호를 훑으며 프리앰블 파형과의 상관을 계산해 가장 큰 지점을 찾고,
        완전히 일치했을 때의 에너지 대비 임계치를 못 넘으면 프레임이
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
