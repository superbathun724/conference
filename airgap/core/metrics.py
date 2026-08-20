"""BER(비트 오류율), 실효 전송률, 유실률 계산.

모든 채널이 이 모듈 하나로 성능을 잰다. 채널마다 계산식이 다르면
"물리적 성질 때문에 성능이 다르다"는 주장이 무너지므로 지표 정의를 여기 한 곳에 고정한다.
BER은 반드시 오류정정(파운틴 부호) 이전, 복조 직후 비트로 잰다.
"""

import numpy as np


def bit_error_rate(sent_bits: np.ndarray, received_bits: np.ndarray) -> float:
    """송신 비트열과 수신 비트열을 같은 위치끼리 비교한 오류 비율.

    두 배열의 길이가 다르면(동기 실패 등) 비교가 무의미하므로 예외를 낸다.
    """
    if len(sent_bits) != len(received_bits):
        raise ValueError(f"비트 길이가 다르다: 송신 {len(sent_bits)}, 수신 {len(received_bits)}")
    if len(sent_bits) == 0:
        return 0.0
    n_errors = int(np.count_nonzero(sent_bits != received_bits))
    return n_errors / len(sent_bits)


def effective_throughput_bps(payload_bytes: int, elapsed_s: float) -> float:
    """복원에 성공한 바이트 수를 걸린 시간으로 나눈 실효 전송률 (bytes/s)."""
    if elapsed_s <= 0:
        raise ValueError(f"경과 시간은 0보다 커야 한다: {elapsed_s}")
    return payload_bytes / elapsed_s


def loss_rate(dropped_count: int, total_count: int) -> float:
    """CRC 실패 등으로 버린 조각 수 ÷ 전체 수신 조각 수."""
    if total_count == 0:
        return 0.0
    return dropped_count / total_count
