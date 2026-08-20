"""acoustic_send.py --out-dir / acoustic_recv.py --in-dir (안드로이드 중계 경로) 테스트.

실제 스피커·마이크 없이도, 폰으로 녹음한 파일이 노트북과 다른 표본율·채널
수로 넘어왔을 때 판정 로직이 올바르게 동작하는지를 확인한다.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.experiments.acoustic_recv import _evaluate, _load_signal


def _sample_frame_bits() -> tuple[np.ndarray, bytes]:
    payload = b"hello airgap"
    frame_bytes = frame.build_frame(seed=1, payload=payload)
    return bytes_to_bits(frame_bytes), payload


def test_evaluate_reports_success_and_payload():
    channel = AcousticFsk(AcousticFskConfig())
    bits, payload = _sample_frame_bits()
    signal = channel.modulate(bits)

    success, decoded_text, reason = _evaluate(channel, signal, expected_payload=payload)

    assert success
    assert decoded_text == payload.decode("utf-8")
    assert reason == ""


def test_evaluate_reports_missing_preamble():
    channel = AcousticFsk(AcousticFskConfig())
    silence = np.zeros(4000)

    success, decoded_text, reason = _evaluate(channel, silence, expected_payload=None)

    assert not success
    assert decoded_text == ""
    assert reason == "프리앰블 미검출"


def test_load_signal_resamples_mismatched_rate(tmp_path):
    """폰이 48kHz로 녹음해도(노트북 설정 44.1kHz) 복조가 그대로 성공해야 한다."""
    channel = AcousticFsk(AcousticFskConfig())
    bits, payload = _sample_frame_bits()
    signal = channel.modulate(bits)

    # 44.1kHz용으로 만든 신호를 48kHz로 실제 리샘플링해 "48kHz로 녹음된 파일"을 흉내낸다.
    recorded_at_48k = resample_poly(signal, 48000, channel.config.sample_rate_hz)
    wav_path = tmp_path / "trial_01.wav"
    sf.write(wav_path, recorded_at_48k, 48000)

    loaded = _load_signal(wav_path, channel.config.sample_rate_hz)
    success, decoded_text, _ = _evaluate(channel, loaded, expected_payload=payload)

    assert success
    assert decoded_text == payload.decode("utf-8")


def test_load_signal_averages_stereo_to_mono(tmp_path):
    channel = AcousticFsk(AcousticFskConfig())
    bits, payload = _sample_frame_bits()
    signal = channel.modulate(bits)

    stereo = np.stack([signal, signal], axis=1)
    wav_path = tmp_path / "trial_01.wav"
    sf.write(wav_path, stereo, channel.config.sample_rate_hz)

    loaded = _load_signal(wav_path, channel.config.sample_rate_hz)
    success, decoded_text, _ = _evaluate(channel, loaded, expected_payload=payload)

    assert loaded.ndim == 1
    assert success
    assert decoded_text == payload.decode("utf-8")
