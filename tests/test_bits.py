import numpy as np

from airgap.core.bits import bits_to_bytes, bytes_to_bits


def test_roundtrip_ascii():
    original = b"hello air-gap"
    bits = bytes_to_bits(original)
    assert len(bits) == len(original) * 8
    assert bits_to_bytes(bits) == original


def test_roundtrip_empty():
    assert bytes_to_bits(b"").size == 0
    assert bits_to_bytes(np.array([], dtype=np.uint8)) == b""


def test_bits_are_msb_first():
    # 0xA5 = 1010 0101
    bits = bytes_to_bits(bytes([0xA5]))
    assert list(bits) == [1, 0, 1, 0, 0, 1, 0, 1]


def test_bits_to_bytes_rejects_non_multiple_of_8():
    try:
        bits_to_bytes(np.array([1, 0, 1], dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("8의 배수가 아닌 비트 길이는 ValueError를 내야 한다")
