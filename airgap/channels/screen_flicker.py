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
    drift_window_s: float = (
        3.0  # 느린 밝기 변동(자동 노출) 제거용 이동 중앙값 창 길이(초). 0이면 끔
    )
    preamble_detect_threshold: float = (
        0.8  # 정규화 상호상관(NCC) 임계치. 잡음만으로도 0.75까지 나오므로 0.8 (2026-09-05)
    )
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
        """신호 → 비트 배열. 프리앰블을 못 찾으면 빈 배열을 돌려준다.

        판정 기준은 고정된 상수가 아니라 **그 시점 주변에서 실제로 관측된 두 준위의
        중간값**이다(_local_threshold). 카메라 자동 노출이 밝기 전체를 눌러 놓거나
        천천히 끌어올려도, 위아래 포락선의 중간을 따라가면 판정이 흔들리지 않는다.
        (2026-09-05 이전에는 (level0+level1)/2 = 0.5라는 상수를 썼고, 실기기에서
        10/10 CRC 불일치로 전부 실패했다. FINDINGS.md 참고)
        """
        start = self._find_preamble_start(signal)
        if start is None:
            return np.array([], dtype=np.uint8)

        threshold = self._local_threshold(signal)

        bits = []
        pos = start
        while pos + self._frames_per_symbol <= len(signal):
            window = signal[pos : pos + self._frames_per_symbol]
            cut = float(np.mean(threshold[pos : pos + self._frames_per_symbol]))
            bits.append(self._decode_symbol(window, cut))
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

    def _decode_symbol(self, window: np.ndarray, threshold: float | None = None) -> int:
        """심볼 구간의 평균 밝기가 기준값보다 높으면 1로 판정한다.

        기준값은 demodulate()가 국소 포락선에서 잰 값을 넘겨준다. 넘겨주지 않으면
        설정의 두 준위 중간값(루프백용)을 쓴다.
        """
        cut = self._midpoint if threshold is None else threshold
        return 1 if float(np.mean(window)) >= cut else 0

    def _local_threshold(self, signal: np.ndarray) -> np.ndarray:
        """각 시점의 판정 기준값 = 주변 창의 (국소 최댓값 + 국소 최솟값) / 2.

        밝기 신호는 두 준위뿐이라, 창 안의 최댓값은 준위 1을, 최솟값은 준위 0을
        따라간다. 카메라 자동 노출이 밝기를 눌러도(진폭 압축) 천천히 끌어올려도
        (드리프트) 두 포락선이 같이 움직이므로 그 중간은 언제나 올바른 기준이다.

        중앙값이나 평균을 쓰면 안 된다. 두 값뿐인 신호의 이동 중앙값은 창 안에
        어느 비트가 많은지에 따라 두 준위 사이를 뛰어다니고, 평균도 비트 밀도에
        끌려간다. 포락선은 비트 밀도와 무관하다.

        창(기본 3초)보다 긴 연속 구간이 있으면 그 안에서 한 준위가 사라져 기준이
        무너진다 — 이것이 이 처리의 대가이며, 창을 길게 잡아 완화한다.
        """
        window = int(round(self.config.fps * self.config.drift_window_s))
        if window < 3 or len(signal) < window:
            return np.full(len(signal), self._midpoint)
        from scipy.ndimage import maximum_filter1d, minimum_filter1d

        upper = maximum_filter1d(signal, size=window, mode="nearest")
        lower = minimum_filter1d(signal, size=window, mode="nearest")
        return (upper + lower) / 2

    def _find_preamble_start(self, signal: np.ndarray) -> int | None:
        """프리앰블 밝기 패턴과 수신 신호를 상관시켜 프레임 시작 지점을 찾는다.

        원리는 음향 채널(acoustic_fsk.py)의 프리앰블 검출과 동일하다 —
        신호를 훑으며 프리앰블 파형과의 상관을 계산해 가장 큰 지점을 찾고,
        완전히 일치했을 때의 에너지 대비 임계치를 못 넘으면 프레임이
        없다고 본다.
        """
        preamble_bits = bytes_to_bits(frame.PREAMBLE)
        preamble_wave = self.modulate(preamble_bits)
        n = len(preamble_wave)
        if len(signal) < n:
            return None

        # 정규화 상호상관(NCC). 두 신호에서 각각 평균을 뺀 뒤 상관시키고 양쪽
        # 에너지로 나눈다. 결과는 -1~1이고, 1이면 모양이 완전히 같다는 뜻이다.
        #
        # 왜 이렇게 하는가. 밝기는 언제나 양수라서, 평균을 빼지 않고 상관시키면
        # 패턴이 맞는 곳이 아니라 **그냥 제일 밝은 구간**에서 최댓값이 나온다.
        # 실기기에서는 카메라 자동 노출 때문에 밝기가 출렁이므로 그 출렁임이
        # 프리앰블 패턴을 눌러 버리고, 엉뚱한 곳을 시작점으로 잡게 된다. 그러면
        # "프리앰블은 검출됐는데 CRC는 항상 불일치"라는 결과가 나온다 — 2026-09-05
        # 실기기 측정이 정확히 그 양상이었다. 소프트웨어 루프백은 밝기가
        # 정확히 두 값뿐이라 이 문제가 드러나지 않았다. (FINDINGS.md 참고)
        template = preamble_wave - preamble_wave.mean()
        template_energy = float(np.sqrt(np.sum(template**2)))
        if template_energy == 0:
            return None

        # 슬라이딩 창마다 평균을 빼야 하므로, 창 합과 제곱합을 누적합으로 구한다.
        ones = np.ones(n)
        window_sum = np.convolve(signal, ones, mode="valid")
        window_sq = np.convolve(signal**2, ones, mode="valid")
        window_var = window_sq - window_sum**2 / n
        window_std = np.sqrt(np.maximum(window_var, 1e-12))

        raw = np.correlate(signal, template, mode="valid")
        ncc = raw / (window_std * template_energy)

        peak_index = int(np.argmax(ncc))
        if ncc[peak_index] < self.config.preamble_detect_threshold:
            return None
        return peak_index
