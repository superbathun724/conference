"""LT 파운틴 부호 테스트. 채널을 전혀 쓰지 않는 순수 알고리즘 계층이라
core/fountain.py만으로 검증할 수 있다 (docs/ARCHITECTURE.md의 계층 구분과 동일한 이유로,
물리층 없이 로직부터 검증한다).
"""

import numpy as np
import pytest

from airgap.core.fountain import (
    LtConfig,
    LtDecoder,
    assemble_chunks,
    encode_droplet,
    robust_soliton_distribution,
    split_into_chunks,
)


def test_split_and_assemble_roundtrip():
    data = b"hello air-gap fountain code"
    chunks = split_into_chunks(data, chunk_size_bytes=8)

    assert all(len(c) == 8 for c in chunks)
    restored = assemble_chunks(dict(enumerate(chunks)), original_length=len(data))
    assert restored == data


def test_split_pads_last_chunk_with_zeros():
    chunks = split_into_chunks(b"12345", chunk_size_bytes=4)

    assert len(chunks) == 2
    assert chunks[-1] == b"5\x00\x00\x00"


def test_robust_soliton_distribution_is_a_valid_probability_table():
    dist = robust_soliton_distribution(k=40, c=0.1, delta=0.05)

    assert len(dist) == 40
    assert np.all(dist >= 0)
    assert dist.sum() == pytest.approx(1.0)


def _generate_case(k: int, chunk_size_bytes: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=k * chunk_size_bytes - 3, dtype=np.uint8).tobytes()
    config = LtConfig(chunk_size_bytes=chunk_size_bytes)
    chunks = split_into_chunks(data, chunk_size_bytes)
    assert len(chunks) == k
    return data, chunks, config


def test_decoder_recovers_original_with_no_loss():
    data, chunks, config = _generate_case(k=20, chunk_size_bytes=16, seed=0)
    k = len(chunks)

    decoder = LtDecoder(k, config)
    droplet_seed = 0
    while not decoder.is_complete and droplet_seed < 10 * k:
        droplet = encode_droplet(chunks, droplet_seed, config)
        decoder.add_droplet(droplet.seed, droplet.payload)
        droplet_seed += 1

    assert decoder.is_complete
    assert decoder.assemble(len(data)) == data


def test_decoder_recovers_with_30_percent_droplet_loss():
    """M2 완료 조건: 방울의 30%를 무작위로 버려도 재전송 없이 복원돼야 한다."""
    data, chunks, config = _generate_case(k=40, chunk_size_bytes=16, seed=1)
    k = len(chunks)

    loss_rng = np.random.default_rng(42)
    decoder = LtDecoder(k, config)
    droplet_seed = 0
    max_generated = 6 * k
    while not decoder.is_complete and droplet_seed < max_generated:
        dropped = loss_rng.random() < 0.3
        if not dropped:
            droplet = encode_droplet(chunks, droplet_seed, config)
            decoder.add_droplet(droplet.seed, droplet.payload)
        droplet_seed += 1

    assert decoder.is_complete, f"{max_generated}개 방울(30% 유실) 안에서 복원 실패"
    assert decoder.assemble(len(data)) == data


def test_encode_droplet_is_deterministic_given_same_seed():
    _, chunks, config = _generate_case(k=10, chunk_size_bytes=16, seed=2)

    first = encode_droplet(chunks, seed=7, config=config)
    second = encode_droplet(chunks, seed=7, config=config)

    assert first == second


def test_duplicate_droplet_does_not_break_decoder():
    data, chunks, config = _generate_case(k=15, chunk_size_bytes=16, seed=3)
    k = len(chunks)

    decoder = LtDecoder(k, config)
    droplet_seed = 0
    while not decoder.is_complete and droplet_seed < 8 * k:
        droplet = encode_droplet(chunks, droplet_seed, config)
        decoder.add_droplet(droplet.seed, droplet.payload)
        decoder.add_droplet(droplet.seed, droplet.payload)  # 같은 방울을 중복 투입
        droplet_seed += 1

    assert decoder.is_complete
    assert decoder.assemble(len(data)) == data


def test_assemble_before_complete_raises():
    _, chunks, config = _generate_case(k=5, chunk_size_bytes=16, seed=4)
    decoder = LtDecoder(len(chunks), config)

    with pytest.raises(RuntimeError):
        decoder.assemble(original_length=10)
