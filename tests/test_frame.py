from airgap.core.frame import build_frame, crc16_ccitt, parse_frame


def test_roundtrip():
    payload = "안녕하세요".encode()
    frame = build_frame(seed=1234, payload=payload)
    parsed = parse_frame(frame)

    assert parsed is not None
    assert parsed.seed == 1234
    assert parsed.payload == payload


def test_roundtrip_empty_payload():
    frame = build_frame(seed=0, payload=b"")
    parsed = parse_frame(frame)

    assert parsed is not None
    assert parsed.payload == b""


def test_bit_flip_is_caught_by_crc():
    frame = build_frame(seed=42, payload=b"attack at dawn")
    corrupted = bytearray(frame)
    corrupted[len(corrupted) // 2] ^= 0b00000001  # 페이로드 중간의 비트 하나를 뒤집는다

    assert parse_frame(bytes(corrupted)) is None


def test_wrong_preamble_is_rejected():
    frame = build_frame(seed=1, payload=b"x")
    corrupted = bytes([0x00, 0x00]) + frame[2:]

    assert parse_frame(corrupted) is None


def test_truncated_frame_is_rejected():
    frame = build_frame(seed=1, payload=b"hello")

    assert parse_frame(frame[:-1]) is None


def test_crc_changes_when_data_changes():
    assert crc16_ccitt(b"A") != crc16_ccitt(b"B")


def test_payload_too_long_raises():
    try:
        build_frame(seed=1, payload=bytes(256))
    except ValueError:
        return
    raise AssertionError("255바이트를 넘는 페이로드는 ValueError를 내야 한다")
