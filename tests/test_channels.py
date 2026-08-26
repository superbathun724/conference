"""채널 파라미터화 테스트.

새 채널을 추가하면 CHANNEL_FACTORIES에 등록만 하면 아래 루프백 테스트가
자동으로 적용된다 (docs/ARCHITECTURE.md '채널 확장 방법' 참고).
"""

import numpy as np
import pytest
import soundfile as sf

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.channels.screen_flicker import ScreenFlicker, ScreenFlickerConfig
from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.core.channel import loopback
from airgap.core.metrics import bit_error_rate

CHANNEL_FACTORIES = {
    "acoustic_fsk": lambda: AcousticFsk(AcousticFskConfig()),
    "screen_flicker": lambda: ScreenFlicker(ScreenFlickerConfig()),
    "screen_qr": lambda: ScreenQr(ScreenQrConfig()),
}


def _sample_frame_bits() -> np.ndarray:
    payload = b"hello airgap"
    frame_bytes = frame.build_frame(seed=1, payload=payload)
    return bytes_to_bits(frame_bytes)


def _trim_to_bytes(bits: np.ndarray) -> np.ndarray:
    """비트 배열을 8의 배수 길이로 잘라낸다 (프레임 뒤 남는 잡음 비트 제거)."""
    return bits[: len(bits) - len(bits) % 8]


@pytest.mark.parametrize("name", CHANNEL_FACTORIES)
def test_loopback_without_noise_recovers_all_bits(name):
    channel = CHANNEL_FACTORIES[name]()
    bits = _sample_frame_bits()

    received = loopback(channel, bits)

    assert len(received) >= len(bits)
    assert np.array_equal(received[: len(bits)], bits)


@pytest.mark.parametrize("name", CHANNEL_FACTORIES)
def test_loopback_recovers_original_frame(name):
    channel = CHANNEL_FACTORIES[name]()
    payload = b"hello airgap"
    bits = bytes_to_bits(frame.build_frame(seed=1, payload=payload))

    received_bits = loopback(channel, bits)
    parsed = frame.parse_frame(bits_to_bytes(_trim_to_bytes(received_bits)))

    assert parsed is not None
    assert parsed.payload == payload


@pytest.mark.parametrize("name", CHANNEL_FACTORIES)
def test_loopback_without_preamble_returns_empty(name):
    channel = CHANNEL_FACTORIES[name]()
    noise = np.random.default_rng(0).normal(0, 0.01, size=4000)

    received = channel.demodulate(noise)

    assert len(received) == 0


def test_ber_rises_as_snr_drops():
    """잡음 없는 루프백은 무오류, SNR을 크게 낮추면 오류율이 뚜렷이 오른다."""
    channel = AcousticFsk(AcousticFskConfig())
    bits = _sample_frame_bits()

    clean = loopback(channel, bits, snr_db=None)
    noisy = loopback(channel, bits, snr_db=-20)

    ber_clean = bit_error_rate(bits, clean[: len(bits)])
    ber_noisy = bit_error_rate(bits, noisy[: len(bits)]) if len(noisy) >= len(bits) else 1.0

    assert ber_clean == 0.0
    assert ber_noisy > ber_clean


def test_wav_roundtrip(tmp_path):
    """modulate() 결과를 WAV로 저장했다가 다시 읽어도 복조가 그대로 성공해야 한다."""
    channel = AcousticFsk(AcousticFskConfig())
    bits = _sample_frame_bits()
    signal = channel.modulate(bits)

    wav_path = tmp_path / "frame.wav"
    sf.write(wav_path, signal, channel.config.sample_rate_hz)
    signal_from_file, sample_rate = sf.read(wav_path)

    assert sample_rate == channel.config.sample_rate_hz
    received = channel.demodulate(signal_from_file)
    assert np.array_equal(received[: len(bits)], bits)
