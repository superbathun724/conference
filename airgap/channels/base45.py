"""바이트열을 QR '영숫자 모드'로 담기 위한 base45 부호화 (RFC 9285).

**왜 필요한가.** QR에 우리 프레임(core/frame.py)의 날바이트를 그대로 넣으면
디코더(zbar)가 그 내용을 텍스트로 보고 문자 인코딩을 멋대로 추정해 바꿔버린다
— 프리앰블 0xF9 0xAB 같은 값이 다른 바이트로 깨진다(FINDINGS.md 2026-08-20).
그래서 ASCII 문자로 감싸서 넣어야 하는데, 처음에는 base64를 썼다.

**왜 base64 대신 base45인가.** QR은 담는 내용의 문자 종류에 따라 다른 모드를
쓴다. 아무 바이트나 담는 '8비트 바이트 모드'는 문자 하나에 8비트를 쓰지만,
숫자·대문자·일부 기호 45종만 쓰는 '영숫자 모드'는 문자 두 개를 11비트에
욱여넣는다(문자당 5.5비트). base64가 만드는 문자에는 소문자가 섞여 있어서
영숫자 모드를 못 쓰고 바이트 모드로 떨어진다:

    base64: 원본 1바이트 -> 4/3 문자 x 8비트   = 10.67비트  (33% 손해)
    base45: 원본 1바이트 -> 3/2 문자 x 5.5비트 =  8.25비트  ( 3% 손해)

즉 문자 수는 base45가 더 많은데도 QR 안에서 차지하는 비트는 더 적다. 같은
조각을 더 작은 QR에 담을 수 있고, QR이 작아지면 모듈(점) 하나가 화면에서
커지므로 카메라 인식에도 유리하다 — 전송률을 올리는 다른 방법들(격자, 표시
시간 단축, 오류정정 하향)이 하나같이 광학적 여유를 깎아먹는 것과 반대다.

**부호화 규칙.** 바이트를 두 개씩 묶어 16비트 수 n으로 보고, 45진법 세 자리
(c, d, e)로 쪼갠다. n = c + d*45 + e*45*45 이고 45^3 = 91125 > 65535이라 세
자리면 언제나 충분하다. 낱개로 남는 마지막 한 바이트는 두 자리로 적는다
(255 < 45^2 = 2025). 자리는 낮은 자리부터 적는다(little-endian).
"""

from __future__ import annotations

# RFC 9285가 정한 45개 문자. 순서가 곧 숫자값이고, 전부 QR 영숫자 모드에
# 들어 있는 문자다. 이 순서를 바꾸면 다른 구현과 호환되지 않는다.
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

_VALUES = {char: value for value, char in enumerate(ALPHABET)}
_BASE = 45


def encode(data: bytes) -> str:
    """바이트열 -> base45 문자열."""
    out: list[str] = []
    # 두 바이트씩 묶어 16비트 수로 만든 뒤 45진법 세 자리로 적는다.
    for i in range(0, len(data) - 1, 2):
        n = data[i] * 256 + data[i + 1]
        n, c = divmod(n, _BASE)
        e, d = divmod(n, _BASE)
        out.append(ALPHABET[c] + ALPHABET[d] + ALPHABET[e])
    # 홀수 길이면 마지막 한 바이트가 남는다. 255는 두 자리로 충분하다.
    if len(data) % 2 == 1:
        d, c = divmod(data[-1], _BASE)
        out.append(ALPHABET[c] + ALPHABET[d])
    return "".join(out)


def decode(text: str) -> bytes:
    """base45 문자열 -> 바이트열.

    카메라가 잘못 읽은 문자열이 여기까지 올라올 수 있으므로, 규칙에 맞지
    않으면 조용히 넘어가지 말고 ValueError를 낸다. 부르는 쪽(screen_qr.py의
    demodulate)이 그것을 잡아 "이 프레임은 버린다"로 처리한다.
    """
    # 길이는 3의 배수(두 바이트씩) 또는 3의 배수 + 2(마지막 한 바이트)여야 한다.
    if len(text) % 3 == 1:
        raise ValueError(f"base45 길이가 규칙에 맞지 않는다: {len(text)}")

    try:
        values = [_VALUES[char] for char in text]
    except KeyError as exc:
        raise ValueError(f"base45에 없는 문자가 있다: {exc.args[0]!r}") from exc

    out = bytearray()
    for i in range(0, len(values) - 2, 3):
        n = values[i] + values[i + 1] * _BASE + values[i + 2] * _BASE * _BASE
        # 45^3(91125)이 65536보다 커서, 세 자리로 16비트를 넘는 수도 적을 수 있다.
        # 그런 문자열은 정상적인 부호화 결과가 아니므로 버린다.
        if n > 0xFFFF:
            raise ValueError(f"base45 세 자리 값이 16비트를 넘는다: {n}")
        out += n.to_bytes(2, "big")
    if len(values) % 3 == 2:
        n = values[-2] + values[-1] * _BASE
        if n > 0xFF:
            raise ValueError(f"base45 마지막 두 자리 값이 8비트를 넘는다: {n}")
        out.append(n)
    return bytes(out)
