"""demo/acoustic_demo.py 테스트. 마이크·스피커 없이 배열과 임시 WAV 파일만으로 검증한다."""

import numpy as np
import soundfile as sf

from airgap.channels.acoustic_fsk import AcousticFsk
from airgap.demo.acoustic_chat import ChatReceiver
from airgap.demo.acoustic_demo import (
    _chunks_from_file,
    build_demo_signal,
    channel_info_line,
    rms,
)

MESSAGES = ["첫 번째", "두 번째"]


def test_build_demo_signal_returns_one_seed_per_message():
    channel = AcousticFsk()

    signal, seeds = build_demo_signal(channel, MESSAGES, gap_s=0.2)

    assert seeds == [1, 2]
    assert len(signal) > 0


def test_build_demo_signal_inserts_silence_between_messages():
    """메시지 사이 침묵이 없으면 두 프레임이 한 버퍼에 섞여 뒤엣것이 묻힌다."""
    channel = AcousticFsk()

    short, _ = build_demo_signal(channel, MESSAGES, gap_s=0.2)
    long, _ = build_demo_signal(channel, MESSAGES, gap_s=1.0)

    extra_samples = len(long) - len(short)
    assert extra_samples == round(1.6 * channel.config.sample_rate_hz)  # 메시지 2개 × 0.8초 차이


def test_rms_is_zero_for_silence_and_positive_for_a_tone():
    assert rms(np.zeros(100)) == 0.0
    assert rms(np.ones(100) * 0.5) == 0.5
    assert rms(np.array([])) == 0.0


def test_channel_info_line_mentions_both_carrier_frequencies():
    line = channel_info_line(AcousticFsk())

    assert "3.0" in line and "4.0" in line


def test_baked_signal_is_recovered_through_the_receive_path(tmp_path):
    """굽기 → WAV 저장 → 조각으로 읽기 → ChatReceiver 순으로 원문이 돌아와야 한다.

    폰이 재생하고 노트북이 녹음하는 실제 경로에서 공기만 뺀 것이다.
    """
    channel = AcousticFsk()
    signal, _ = build_demo_signal(channel, MESSAGES, gap_s=1.0)
    wav_path = tmp_path / "demo.wav"
    sf.write(wav_path, signal, channel.config.sample_rate_hz)

    receiver = ChatReceiver(channel)
    received = []
    for index, chunk in enumerate(
        _chunks_from_file(wav_path, channel.config.sample_rate_hz, chunk_s=0.5, realtime=False)
    ):
        receiver.append(chunk)
        if index % 2 == 0:  # 실제 데모처럼 조각마다가 아니라 가끔만 복조를 시도한다
            text = receiver.try_decode()
            if text is not None:
                received.append(text)
                receiver.reset_buffer()  # 데모 루프와 같은 처리
    text = receiver.try_decode()
    if text is not None:
        received.append(text)

    assert received == MESSAGES


def test_second_message_is_lost_without_reset_buffer(tmp_path):
    """reset_buffer()가 왜 필요한지 고정해두는 테스트 — 안 부르면 두 번째를 놓친다."""
    channel = AcousticFsk()
    signal, _ = build_demo_signal(channel, MESSAGES, gap_s=1.0)
    wav_path = tmp_path / "demo.wav"
    sf.write(wav_path, signal, channel.config.sample_rate_hz)

    receiver = ChatReceiver(channel)
    received = []
    for chunk in _chunks_from_file(
        wav_path, channel.config.sample_rate_hz, chunk_s=0.5, realtime=False
    ):
        receiver.append(chunk)
        text = receiver.try_decode()
        if text is not None:
            received.append(text)  # 일부러 reset_buffer()를 부르지 않는다

    assert received == MESSAGES[:1]


def test_file_chunks_are_resampled_to_the_channel_rate(tmp_path):
    """폰 녹음이 48kHz여도 채널 표본율(44.1kHz)에 맞춰져 나와야 한다."""
    path = tmp_path / "phone.wav"
    sf.write(path, np.zeros(48000), 48000)

    chunks = list(_chunks_from_file(path, 44100, chunk_s=1.0, realtime=False))

    total = sum(len(c) for c in chunks)
    assert abs(total - 44100) < 100  # 1초 분량으로 리샘플링됐다
