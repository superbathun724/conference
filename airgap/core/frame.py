"""프레임 조립/해체: [프리앰블 2B][시드 2B][길이 1B][페이로드][CRC16 2B]

CRC(순환 중복 검사, Cyclic Redundancy Check)란: 데이터를 다항식 나눗셈을 하듯
비트 단위로 처리해 나머지를 남기는 오류 검출 코드다. 송신 측이 데이터로 나머지를
계산해 뒤에 붙이고, 수신 측이 받은 데이터로 같은 계산을 다시 해서 값이 같은지
확인한다. 값이 다르면 전송 중 최소 한 비트가 뒤집혔다는 뜻이다. 여기서는
CRC-16/CCITT-FALSE(다항식 0x1021, 초기값 0xFFFF)를 그대로 손으로 구현한다.

프리앰블은 프레임의 시작을 알리는 고정 비트 패턴이다. 바커 부호(Barker code)처럼
자기상관이 뾰족한 패턴을 쓰면 잡음 속에서도 "여기서부터 프레임이 시작한다"는
지점을 찾기 쉽다. 실제 잡음 낀 신호에서 이 패턴을 찾는 상관 검출은 M1(음향 채널)
에서 구현하고, 이 모듈은 이미 위치가 잡힌 프레임의 조립/해체만 담당한다.
"""

from dataclasses import dataclass

# 바커 부호 13(자기상관이 뾰족한 이진 패턴)에 3비트를 덧붙여 16비트로 맞춘 고정 프리앰블.
PREAMBLE = bytes([0xF9, 0xAB])  # 0b11111001_10101011

_SEED_LEN = 2
_LENGTH_LEN = 1
_CRC_LEN = 2
_HEADER_LEN = len(PREAMBLE) + _SEED_LEN + _LENGTH_LEN
_MIN_FRAME_LEN = _HEADER_LEN + _CRC_LEN


@dataclass(frozen=True)
class ParsedFrame:
    seed: int
    payload: bytes


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE. data 각 바이트를 최상위 비트부터 하나씩 밀어 넣는다."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_frame(seed: int, payload: bytes) -> bytes:
    """시드와 페이로드로 프레임 전체 바이트열을 조립한다."""
    if not 0 <= seed <= 0xFFFF:
        raise ValueError(f"시드는 0~65535 범위여야 한다: {seed}")
    if len(payload) > 255:
        raise ValueError(f"페이로드는 255바이트를 넘을 수 없다: {len(payload)}")

    body = seed.to_bytes(_SEED_LEN, "big") + len(payload).to_bytes(_LENGTH_LEN, "big") + payload
    crc = crc16_ccitt(body)
    return PREAMBLE + body + crc.to_bytes(_CRC_LEN, "big")


def parse_frame(data: bytes) -> ParsedFrame | None:
    """프레임 바이트열을 시드/페이로드로 해체한다.

    프리앰블이 맞지 않거나, 길이가 모자라거나, CRC가 어긋나면
    (비트가 뒤집혔다는 뜻이므로) None을 돌려준다. 이 프레임은 버린다.
    """
    if len(data) < _MIN_FRAME_LEN or data[: len(PREAMBLE)] != PREAMBLE:
        return None

    seed = int.from_bytes(data[len(PREAMBLE) : len(PREAMBLE) + _SEED_LEN], "big")
    length = data[len(PREAMBLE) + _SEED_LEN]
    total_len = _HEADER_LEN + length + _CRC_LEN
    if len(data) < total_len:
        return None

    body = data[len(PREAMBLE) : _HEADER_LEN + length]
    payload = data[_HEADER_LEN : _HEADER_LEN + length]
    crc_received = int.from_bytes(data[_HEADER_LEN + length : total_len], "big")

    if crc16_ccitt(body) != crc_received:
        return None
    return ParsedFrame(seed=seed, payload=payload)
