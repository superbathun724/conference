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


def test_screen_qr_survives_a_long_run_of_zero_bytes():
    """0으로 채워진 페이로드도 QR로 만들어지고 그대로 되돌아와야 한다.

    파운틴 조각의 마지막 하나는 원본 길이를 맞추려고 0으로 채워지므로, 이건
    드문 경우가 아니라 파일을 보낼 때 거의 항상 생기는 경우다. 스크램블
    (screen_qr.py의 _scramble)이 빠지면 여기서 QR 생성이 실패한다.
    """
    channel = ScreenQr(ScreenQrConfig())
    payload = b"AIRGAP" + bytes(192)  # 뒤쪽 192바이트가 전부 0
    bits = bytes_to_bits(frame.build_frame(seed=0, payload=payload))

    received = channel.demodulate(channel.modulate(bits))
    parsed = frame.parse_frame(bits_to_bytes(_trim_to_bytes(received)))

    assert parsed is not None
    assert parsed.payload == payload


def test_screen_qr_still_reads_legacy_base64_symbols():
    """2026-09-03 이전에 base64로 찍어둔 측정 원본을 지금 코드로도 읽을 수 있어야 한다.

    CLAUDE.md 규칙 3(측정 원본은 손대지 않는다)의 실질적 조건이다 — 원본이
    남아 있어도 지금 코드가 못 읽으면 보존한 의미가 없다.
    """
    import base64

    import numpy as np
    import qrcode

    frame_bytes = frame.build_frame(seed=1, payload=b"hello airgap")
    qr = qrcode.QRCode(box_size=6, border=4)
    qr.add_data(base64.b64encode(frame_bytes))  # 예전 방식 그대로
    qr.make(fit=True)
    legacy_image = np.array(
        qr.make_image(fill_color="black", back_color="white").convert("L"), dtype=np.uint8
    )

    received = ScreenQr(ScreenQrConfig()).demodulate(legacy_image)

    assert frame.parse_frame(bits_to_bytes(_trim_to_bytes(received))) is not None


# ---- 2026-09-05 실기기 실패 원인에 대한 회귀 테스트 ----
# 세 채널 중 둘이 실기기에서 0/10이었다. 원인을 시뮬레이션으로 재현하고 고쳤으므로,
# 그 조건들을 여기 고정해 둔다. 이 테스트가 깨지면 실기기 실패가 되돌아온 것이다.


def _acoustic_frame_bits():
    return bytes_to_bits(frame.build_frame(seed=1, payload=b"AIRGAP regression"))


def _recovers(channel, received_bits, expected_payload=b"AIRGAP regression") -> bool:
    bits = _trim_to_bytes(received_bits)
    if len(bits) == 0:
        return False
    parsed = frame.parse_frame(bits_to_bytes(bits))
    return parsed is not None and parsed.payload == expected_payload


def test_acoustic_survives_clock_mismatch_between_devices():
    """재생 기기와 녹음 기기의 시계가 0.5% 어긋나도 복원돼야 한다.

    2026-09-05 1m 측정에서 '프리앰블은 검출, CRC는 불일치'가 6/10이었고, 시뮬레이션
    에서 0.2%의 시계 오차만으로 그 양상이 그대로 재현됐다. 심볼 주기 추정
    (_estimate_symbol_period)이 빠지면 여기서 실패한다.
    """
    channel = AcousticFsk(AcousticFskConfig())
    clean = channel.modulate(_acoustic_frame_bits())
    fs = channel.config.sample_rate_hz
    stretched_len = int(round(len(clean) * 1.005))
    stretched = np.interp(
        np.linspace(0, len(clean) - 1, stretched_len), np.arange(len(clean)), clean
    )
    signal = np.concatenate([np.zeros(fs // 2), 0.4 * stretched, np.zeros(fs // 2)])

    assert _recovers(channel, channel.demodulate(signal))


def test_acoustic_survives_low_amplitude():
    """수신 진폭이 0.15배로 작아져도 프리앰블을 찾아야 한다.

    8월 10cm 측정에서 시행별 진폭이 0.24~0.92배로 변동했고 '프리앰블 미검출'이
    5/10이었다. 예전 검출기는 상관값을 송신 파형 에너지로만 나눠 진폭이 작으면
    임계값에 못 미쳤다. 톤 에너지 비율 기반 검출은 진폭에 무관하다.
    """
    channel = AcousticFsk(AcousticFskConfig())
    clean = channel.modulate(_acoustic_frame_bits())
    fs = channel.config.sample_rate_hz
    rng = np.random.default_rng(5)
    signal = np.concatenate([np.zeros(fs // 2), 0.15 * clean, np.zeros(fs // 2)])
    signal = signal + rng.normal(0, 0.005, len(signal))

    assert _recovers(channel, channel.demodulate(signal))


def test_flicker_survives_auto_exposure_compression_and_drift():
    """카메라 자동 노출이 진폭을 20%로 누르고 느린 밝기 변동을 얹어도 복원돼야 한다.

    2026-09-05 30cm 측정에서 10/10 CRC 불일치였다. 두 가지 결함이 겹쳐 있었다 —
    (1) 양수 신호끼리의 상관은 패턴이 아니라 밝은 구간에서 최대가 되어 엉뚱한
    시작점을 잡았고, (2) 고정 기준값 0.5는 눌린 진폭을 판정하지 못했다.
    """
    channel = ScreenFlicker(ScreenFlickerConfig())
    bits = bytes_to_bits(frame.build_frame(seed=1, payload=b"AIRGAP regression"))
    clean = channel.modulate(bits)
    t = np.linspace(0, 1, len(clean))
    compressed = 0.2 * (clean - 0.5) + 0.5 + 0.15 * np.sin(2 * np.pi * 0.7 * t)
    rng = np.random.default_rng(9)
    signal = np.concatenate([np.full(40, compressed[0]), compressed, np.full(40, compressed[-1])])
    signal = signal + rng.normal(0, 0.02, len(signal))

    assert _recovers(channel, channel.demodulate(signal))
