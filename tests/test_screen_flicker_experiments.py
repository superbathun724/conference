"""screen_flicker_send.py / screen_flicker_recv.py --in-dir (안드로이드 중계 경로) 테스트.

실제 카메라 없이도, 폰으로 찍은 동영상 파일이 노트북 설정과 다른 fps로
넘어왔을 때 판정 로직이 올바르게 동작하는지를 확인한다
(tests/test_acoustic_experiments.py의 표본율 리샘플링 테스트와 같은 목적,
매질만 소리에서 화면으로 바뀐 것).
"""

from dataclasses import replace

import cv2
import numpy as np

from airgap.channels.screen_flicker import ScreenFlicker, ScreenFlickerConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.experiments.screen_flicker_recv import _evaluate, _sample_video_file

_FRAME_SIZE_PX = (64, 48)  # 테스트용 축소 해상도 — 내용 자체는 무관하고 속도만 중요하다


def _sample_frame_bits() -> tuple[np.ndarray, bytes]:
    payload = b"hello airgap"
    frame_bytes = frame.build_frame(seed=1, payload=payload)
    return bytes_to_bits(frame_bytes), payload


def _write_video(path, signal: np.ndarray, fps: float) -> None:
    """밝기값 배열(0~1)을 그레이스케일 단색 프레임 동영상으로 써낸다."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = _FRAME_SIZE_PX
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), False)
    try:
        for level in signal:
            gray_value = int(round(np.clip(level, 0.0, 1.0) * 255))
            writer.write(np.full((height, width), gray_value, dtype=np.uint8))
    finally:
        writer.release()


def test_evaluate_reports_success_and_payload():
    channel = ScreenFlicker(ScreenFlickerConfig())
    bits, payload = _sample_frame_bits()
    signal = channel.modulate(bits)

    success, decoded_text, reason = _evaluate(channel, signal, expected_payload=payload)

    assert success
    assert decoded_text == payload.decode("utf-8")
    assert reason == ""


def test_evaluate_reports_missing_preamble():
    channel = ScreenFlicker(ScreenFlickerConfig())
    darkness = np.zeros(200)

    success, decoded_text, reason = _evaluate(channel, darkness, expected_payload=None)

    assert not success
    assert decoded_text == ""
    assert reason == "프리앰블 미검출"


def test_sample_video_file_roundtrip_at_matching_fps(tmp_path):
    """송신 fps와 동영상 fps가 같으면 동영상 경유로도 그대로 복조돼야 한다."""
    config = ScreenFlickerConfig()
    channel = ScreenFlicker(config)
    bits, payload = _sample_frame_bits()
    signal = channel.modulate(bits)

    video_path = tmp_path / "trial_01.mp4"
    _write_video(video_path, signal, config.fps)

    samples, video_fps = _sample_video_file(video_path, config)
    trial_channel = ScreenFlicker(replace(config, fps=video_fps))
    success, decoded_text, _ = _evaluate(trial_channel, samples, expected_payload=payload)

    assert success
    assert decoded_text == payload.decode("utf-8")


def test_sample_video_file_roundtrip_at_different_fps(tmp_path):
    """폰 동영상 fps(예: 24)가 채널 설정 fps(30)와 달라도, 실제 fps로 복조하면 성공해야 한다."""
    config = ScreenFlickerConfig()
    channel = ScreenFlicker(config)
    bits, payload = _sample_frame_bits()
    signal = channel.modulate(bits)

    recorded_fps = 24.0
    # 심볼 하나당 프레임 수가 달라지도록, 신호를 recorded_fps 기준 프레임 수로 다시 늘어놓는다
    # (실제 카메라가 config.fps가 아니라 자기 fps로 프레임을 찍는 상황을 흉내낸다).
    frames_per_symbol_at_recorded_fps = max(
        1, round(recorded_fps * config.symbol_duration_ms / 1000)
    )
    symbols = signal[:: channel._frames_per_symbol]  # noqa: SLF001 (테스트 전용 내부 접근)
    resampled_signal = np.repeat(symbols, frames_per_symbol_at_recorded_fps)

    video_path = tmp_path / "trial_01.mp4"
    _write_video(video_path, resampled_signal, recorded_fps)

    samples, video_fps = _sample_video_file(video_path, config)
    assert video_fps == recorded_fps

    trial_channel = ScreenFlicker(replace(config, fps=video_fps))
    success, decoded_text, _ = _evaluate(trial_channel, samples, expected_payload=payload)

    assert success
    assert decoded_text == payload.decode("utf-8")
