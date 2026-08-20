import numpy as np
import pytest

from airgap.core.metrics import bit_error_rate, effective_throughput_bps, loss_rate


def test_bit_error_rate_no_errors():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    assert bit_error_rate(bits, bits) == 0.0


def test_bit_error_rate_all_flipped():
    sent = np.array([1, 0, 1, 0], dtype=np.uint8)
    received = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert bit_error_rate(sent, received) == 1.0


def test_bit_error_rate_partial():
    sent = np.array([1, 1, 1, 1], dtype=np.uint8)
    received = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert bit_error_rate(sent, received) == 0.5


def test_bit_error_rate_length_mismatch_raises():
    with pytest.raises(ValueError):
        bit_error_rate(np.array([1, 0]), np.array([1, 0, 1]))


def test_effective_throughput():
    assert effective_throughput_bps(payload_bytes=100, elapsed_s=2.0) == 50.0


def test_effective_throughput_rejects_zero_time():
    with pytest.raises(ValueError):
        effective_throughput_bps(payload_bytes=100, elapsed_s=0.0)


def test_loss_rate():
    assert loss_rate(dropped_count=3, total_count=10) == 0.3


def test_loss_rate_zero_total_is_zero():
    assert loss_rate(dropped_count=0, total_count=0) == 0.0
