import numpy as np
import pytest

from airgap.core.channel import Channel, ChannelCaps, add_awgn, loopback


class _LoopbackChannel(Channel):
    """channel.py 자체를 테스트하기 위한 최소 더미 구현. 실제 채널이 아니다."""

    caps = ChannelCaps(
        name="dummy",
        medium="air",
        wave_type="longitudinal",
        coding_axes=("time",),
        directional=False,
        penetrates_opaque=False,
        human_perceptible=True,
        nominal_bps=1.0,
    )

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        return bits.astype(np.float64)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        return (signal > 0.5).astype(np.uint8)

    def emit(self, signal: np.ndarray) -> None:
        raise NotImplementedError("더미 채널은 실제 장치가 없다")

    def capture(self, duration_s: float) -> np.ndarray:
        raise NotImplementedError("더미 채널은 실제 장치가 없다")


def test_channel_is_abstract():
    with pytest.raises(TypeError):
        Channel()


def test_channel_caps_is_frozen():
    with pytest.raises(Exception):
        _LoopbackChannel.caps.name = "changed"


def test_loopback_without_noise_is_exact():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    received = loopback(_LoopbackChannel(), bits)
    assert np.array_equal(received, bits)


def test_add_awgn_is_reproducible_with_same_seed():
    signal = np.ones(1000)
    noisy_a = add_awgn(signal, snr_db=10, seed=7)
    noisy_b = add_awgn(signal, snr_db=10, seed=7)
    assert np.array_equal(noisy_a, noisy_b)


def test_add_awgn_changes_signal():
    signal = np.ones(1000)
    noisy = add_awgn(signal, snr_db=0, seed=1)
    assert not np.array_equal(noisy, signal)
