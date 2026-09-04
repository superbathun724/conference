"""demo/manifest.py 테스트. 안내표는 수신측이 무엇을 받는지 아는 유일한 통로라 꼼꼼히 본다."""

import math

from airgap.core.fountain import split_into_chunks
from airgap.demo.manifest import (
    MANIFEST_SEED,
    Manifest,
    build_manifest,
    parse_manifest,
    safe_name,
)


def _manifest(**overrides) -> Manifest:
    values = {
        "name": "cat.png",
        "size_bytes": 10240,
        "chunk_size_bytes": 192,
        "c": 0.1,
        "delta": 0.05,
    }
    values.update(overrides)
    return Manifest(**values)


def test_manifest_survives_a_round_trip():
    manifest = _manifest()

    assert parse_manifest(build_manifest(manifest)) == manifest


def test_manifest_fits_in_one_frame():
    """프레임 페이로드는 255바이트가 상한(core/frame.py의 길이 칸이 8비트)이다."""
    manifest = _manifest(name="아주" * 40 + ".png")

    assert len(build_manifest(manifest)) <= 255


def test_manifest_k_matches_split_into_chunks():
    """수신측이 계산한 K가 송신측이 실제로 나눈 조각 수와 달라지면 복원이 영영 안 끝난다."""
    for size in [1, 191, 192, 193, 10240, 10241]:
        manifest = _manifest(size_bytes=size)
        expected = len(split_into_chunks(bytes(size), manifest.chunk_size_bytes))
        assert manifest.k == expected, f"size={size}"


def test_manifest_seed_is_reserved_zero():
    """방울 시드는 1부터 매기므로 0이 안내표 전용으로 남는다."""
    assert MANIFEST_SEED == 0


def test_parse_rejects_garbage_instead_of_raising():
    """반쯤 찍힌 프레임 때문에 데모가 멈추면 안 된다 — 조용히 None."""
    assert parse_manifest(b"garbage") is None
    assert parse_manifest(b"AIRGAP1|a|b|c|d|e") is None  # 숫자 자리에 글자
    assert parse_manifest(b"OTHER1|cat.png|10|8|0.1|0.05") is None  # 규격이 다름
    assert parse_manifest(b"AIRGAP1|cat.png|10|8") is None  # 칸 수가 모자람
    assert parse_manifest(b"\xff\xfe") is None  # UTF-8이 아님


def test_parse_rejects_impossible_sizes():
    assert parse_manifest(b"AIRGAP1|cat.png|-1|192|0.1|0.05") is None
    assert parse_manifest(b"AIRGAP1|cat.png|10|0|0.1|0.05") is None


def test_safe_name_strips_directories():
    """받은 이름을 그대로 저장 경로에 쓰면 엉뚱한 곳에 파일을 쓸 수 있다."""
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("C:\\Windows\\x.png").endswith("x.png")


def test_safe_name_replaces_the_separator():
    """파일명에 구분자가 들어가면 안내표 자체가 깨진다."""
    assert "|" not in safe_name("a|b.png")


def test_safe_name_is_never_empty():
    assert safe_name("   ") == "payload.bin"


def test_long_names_are_truncated_not_rejected():
    name = safe_name("가" * 200 + ".png")

    assert len(name.encode("utf-8")) <= 64
    assert math.ceil(len(name)) > 0
