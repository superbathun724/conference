"""바이트열과 비트 배열(0/1) 사이를 변환한다.

프레임층(frame.py)은 바이트 단위로 동작하지만, 물리층(channels/*.py)은
반송 주파수 하나에 비트 하나를 실어야 하므로 0/1이 낱개로 늘어선 배열이 필요하다.
이 모듈이 그 경계를 잇는다.

바이트 하나(8비트)는 항상 MSB(최상위 비트)부터 순서대로 풀고 합친다.
송신과 수신이 같은 순서를 쓰지 않으면 프레임 전체가 어긋나므로 이 규칙은 고정한다.
"""

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    """바이트열 → 0/1 비트 배열 (MSB first). 길이는 len(data) * 8."""
    byte_array = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(byte_array)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """0/1 비트 배열 → 바이트열. 길이가 8의 배수여야 한다."""
    if len(bits) % 8 != 0:
        raise ValueError(f"비트 길이가 8의 배수가 아니다: {len(bits)}")
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes()
