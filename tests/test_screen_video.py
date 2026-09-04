"""화면 채널을 동영상 파일로 써내는 경로(안드로이드 중계) 테스트.

핵심 확인 사항은 하나다 — 폰으로 재생·녹화할 동영상이 **판정 가능한 형태로**
만들어지는가. 실제 카메라를 거치지 않은 이상적 조건이지만, 파일 형식이나
프레임 반복 계산이 어긋나면 여기서 걸린다.
"""

import cv2
import numpy as np
import pytest

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.experiments import qr_grid, screen_video


def test_frames_per_screen_matches_display_time():
    """30fps 동영상에서 500ms를 유지하려면 같은 화면을 15프레임 써야 한다."""
    assert screen_video.frames_per_screen(500.0, 30.0) == 15
    assert screen_video.frames_per_screen(100.0, 30.0) == 3
    # 프레임 간격보다 짧은 표시 시간도 최소 한 장은 나와야 한다(화면이 사라지면 안 된다).
    assert screen_video.frames_per_screen(5.0, 30.0) == 1


def test_written_video_is_readable_and_has_expected_length(tmp_path):
    screens = [np.full((120, 120), v, dtype=np.uint8) for v in (0, 255, 0)]
    path = tmp_path / "trial_01.mp4"

    written = screen_video.write_video(screens, path, display_ms=100.0, video_fps=30.0)

    assert written == 3 * 3
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 30
    finally:
        capture.release()


def test_screens_of_different_sizes_share_one_canvas(tmp_path):
    """QR 크기가 화면마다 달라도 동영상 프레임 크기는 하나로 고정돼야 한다.

    재생 중 그림 크기가 들쭉날쭉하면 카메라가 초점을 계속 다시 잡는다.
    """
    screens = [np.zeros((80, 80), np.uint8), np.zeros((140, 140), np.uint8)]

    screen_video.write_video(screens, tmp_path / "t.mp4", display_ms=100.0, video_fps=30.0)

    capture = cv2.VideoCapture(str(tmp_path / "t.mp4"))
    try:
        sizes = set()
        while True:
            ok, image = capture.read()
            if not ok:
                break
            sizes.add(image.shape[:2])
        assert sizes == {(140, 140)}
    finally:
        capture.release()


@pytest.mark.parametrize("grid", [1, 2])
def test_droplets_survive_the_video_roundtrip(tmp_path, grid):
    """동영상으로 써낸 QR을 되읽어 방울이 그대로 나와야 한다.

    폰 재생 → 폰 녹화 경로의 이상적 상한선에 해당한다. 여기서 실패하면
    실기기에서는 볼 것도 없다.
    """
    channel = ScreenQr(ScreenQrConfig())
    seeds = list(range(1, qr_grid.droplets_per_screen(grid) + 1))
    images = [
        channel.modulate(
            bytes_to_bits(
                frame.build_frame(seed=s, payload=bytes(((i * s + 7) % 255) + 1 for i in range(48)))
            )
        )
        for s in seeds
    ]
    screen = qr_grid.tile(images, grid) if grid > 1 else images[0]

    path = tmp_path / "trial_01.mp4"
    screen_video.write_video([screen], path, display_ms=200.0, video_fps=30.0)

    capture = cv2.VideoCapture(str(path))
    try:
        ok, image = capture.read()
        assert ok
    finally:
        capture.release()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parsed = [frame.parse_frame(bits_to_bytes(b)) for b in channel.demodulate_all(gray)]
    assert sorted(p.seed for p in parsed if p) == seeds


def test_write_video_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        screen_video.write_video([], tmp_path / "t.mp4", display_ms=100.0)
