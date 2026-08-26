"""demo/acoustic_chat.py의 ChatReceiver 테스트.

실제 마이크·스레드 없이도, 오디오 배열을 직접 feed()해서 판정 로직(새
메시지 인식, 자기 반향 걸러내기, 시드 충돌 처리)을 확인할 수 있다.
"""

import numpy as np

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.demo.acoustic_chat import ChatReceiver


def _signal_for(channel: AcousticFsk, seed: int, payload: bytes) -> np.ndarray:
    frame_bytes = frame.build_frame(seed=seed, payload=payload)
    bits = bytes_to_bits(frame_bytes)
    return channel.modulate(bits)


def test_feed_returns_new_incoming_message():
    channel = AcousticFsk(AcousticFskConfig())
    receiver = ChatReceiver(channel)
    signal = _signal_for(channel, seed=1, payload=b"hello")

    text = receiver.feed(signal)

    assert text == "hello"


def test_own_message_is_filtered_as_echo():
    """emit() 전에 mark_own()으로 등록해둔 (시드, 내용)은 다시 들려도 '수신'으로 안 뜬다."""
    channel = AcousticFsk(AcousticFskConfig())
    receiver = ChatReceiver(channel)
    signal = _signal_for(channel, seed=1, payload=b"hello")

    receiver.mark_own(seed=1, payload=b"hello")
    text = receiver.feed(signal)

    assert text is None


def test_same_message_is_not_reported_twice():
    """버퍼가 겹쳐 같은 프레임이 여러 번 디코딩돼도 한 번만 보고한다."""
    channel = AcousticFsk(AcousticFskConfig())
    receiver = ChatReceiver(channel)
    signal = _signal_for(channel, seed=1, payload=b"hello")

    first = receiver.feed(signal)
    second = receiver.feed(np.zeros(100))  # 조각을 더 흘려보내도 버퍼엔 여전히 같은 프레임뿐

    assert first == "hello"
    assert second is None


def test_seed_collision_with_different_payload_is_not_confused():
    """두 기기가 각자 1부터 시드를 매기다 같은 시드를 쓰더라도, 내용이 다르면 별개로 봐야 한다."""
    channel = AcousticFsk(AcousticFskConfig())
    receiver = ChatReceiver(channel)

    receiver.mark_own(seed=1, payload=b"my message")
    incoming = _signal_for(channel, seed=1, payload=b"their message")

    text = receiver.feed(incoming)

    assert text == "their message"
