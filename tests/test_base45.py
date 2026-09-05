"""base45(RFC 9285) 부호화 테스트."""

import pytest

from airgap.channels import base45
from airgap.core import frame


@pytest.mark.parametrize(
    ("raw", "encoded"),
    [
        (b"AB", "BB8"),
        (b"Hello!!", "%69 VD92EX0"),
        (b"base-45", "UJCLQE7W581"),
        (b"ietf!", "QED8WEX0"),
    ],
)
def test_rfc9285_vectors(raw, encoded):
    """RFC 9285 부록의 예제와 정확히 같은 결과가 나와야 한다.

    우리끼리 왕복만 맞으면 통과하는 자체 규격이 아니라 표준을 따랐다는
    확인이다 — 나중에 다른 구현(폰 앱 등)과 붙일 때 이게 근거가 된다.
    """
    assert base45.encode(raw) == encoded
    assert base45.decode(encoded) == raw


@pytest.mark.parametrize("size", [0, 1, 2, 3, 16, 17, 192, 255])
def test_roundtrip_various_lengths(size):
    """홀수·짝수 길이 모두 왕복해야 한다(홀수면 마지막 한 바이트가 두 자리로 남는다)."""
    raw = bytes((i * 37 + 11) % 256 for i in range(size))
    assert base45.decode(base45.encode(raw)) == raw


def test_roundtrip_preserves_frame_bytes():
    """0x80 이상 바이트(프리앰블·CRC)가 그대로 살아남아야 한다 — base64를 쓴 원래 이유."""
    original = frame.build_frame(seed=0xF9AB, payload=b"\xff\x80\x00\xab hello")
    restored = base45.decode(base45.encode(original))

    assert restored == original
    assert frame.parse_frame(restored) is not None


@pytest.mark.parametrize(
    "bad",
    [
        "A",  # 길이가 3의 배수도, 3의 배수+2도 아니다
        "ABCD",
        "abc",  # 소문자는 base45 문자집합에 없다
        "GGW",  # 세 자리 값이 65536 -> 16비트를 넘는다
        ":::",
    ],
)
def test_decode_rejects_malformed_input(bad):
    """카메라가 잘못 읽은 문자열은 조용히 통과시키지 말고 거부해야 한다."""
    with pytest.raises(ValueError):
        base45.decode(bad)
