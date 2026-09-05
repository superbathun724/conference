"""QR 격자 배치(축 A 실험 변인) 테스트.

핵심 확인 사항은 하나다 — 화면에 QR을 n x n으로 띄웠을 때, 수신측이 이미지를
잘라 나누지 않고도 그 안의 방울을 **전부** 되찾을 수 있는가.
"""

import numpy as np
import pytest

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.experiments import qr_grid


def _droplet_image(channel: ScreenQr, seed: int) -> np.ndarray:
    """시드마다 내용이 다른 프레임 하나를 QR 이미지로 만든다."""
    payload = bytes(((i * seed + 7) % 255) + 1 for i in range(64))
    frame_bytes = frame.build_frame(seed=seed, payload=payload)
    return channel.modulate(bytes_to_bits(frame_bytes))


def test_tile_shape_is_grid_times_cell():
    channel = ScreenQr(ScreenQrConfig())
    images = [_droplet_image(channel, seed) for seed in range(1, 5)]

    tiled = qr_grid.tile(images, grid=2)

    cell_h = max(img.shape[0] for img in images)
    cell_w = max(img.shape[1] for img in images)
    assert tiled.shape == (cell_h * 2, cell_w * 2)


def test_missing_cells_are_left_white():
    """이미지가 칸 수보다 적으면 남는 칸은 흰색이어야 한다(QR이 없는 것뿐)."""
    channel = ScreenQr(ScreenQrConfig())
    images = [_droplet_image(channel, 1)]

    tiled = qr_grid.tile(images, grid=2)

    cell_h, cell_w = images[0].shape
    assert np.all(tiled[:, cell_w:] == qr_grid.WHITE)
    assert np.all(tiled[cell_h:, :] == qr_grid.WHITE)


@pytest.mark.parametrize("grid", [1, 2, 3])
def test_all_droplets_survive_the_grid_roundtrip(grid):
    """n x n으로 띄운 방울이 하나도 빠짐없이 되돌아와야 한다."""
    channel = ScreenQr(ScreenQrConfig())
    seeds = list(range(1, qr_grid.droplets_per_screen(grid) + 1))
    images = [_droplet_image(channel, seed) for seed in seeds]

    tiled = qr_grid.tile(images, grid)
    recovered = channel.demodulate_all(tiled)
    parsed = [frame.parse_frame(bits_to_bytes(bits)) for bits in recovered]

    assert len(parsed) == len(seeds)
    assert sorted(p.seed for p in parsed) == seeds


def test_demodulate_returns_one_frame_from_a_grid():
    """Channel 인터페이스(신호 하나 -> 프레임 하나)는 그대로여야 한다."""
    channel = ScreenQr(ScreenQrConfig())
    images = [_droplet_image(channel, seed) for seed in (11, 22, 33, 44)]

    tiled = qr_grid.tile(images, grid=2)
    bits = channel.demodulate(tiled)

    parsed = frame.parse_frame(bits_to_bytes(bits))
    assert parsed is not None
    assert parsed.seed in (11, 22, 33, 44)


def test_tile_rejects_too_many_images():
    channel = ScreenQr(ScreenQrConfig())
    images = [_droplet_image(channel, seed) for seed in range(1, 6)]

    with pytest.raises(ValueError):
        qr_grid.tile(images, grid=2)
