"""demo/screen_qr_demo.py 테스트. 카메라 없이 합성 QR 이미지만으로 검증한다."""

import numpy as np

from airgap.channels.screen_qr import ScreenQr
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.core.fountain import LtConfig, encode_droplet, split_into_chunks
from airgap.demo.manifest import MANIFEST_SEED, Manifest, build_manifest
from airgap.demo.screen_qr_demo import (
    QrReceiveSession,
    RateMeter,
    frame_plan,
    goodput_kbps,
)

PAYLOAD = bytes(range(256)) * 8  # 2KB — 이미지 파일 대역의 축소판
CHUNK_SIZE = 192


def _setup():
    config = LtConfig(chunk_size_bytes=CHUNK_SIZE)
    chunks = split_into_chunks(PAYLOAD, CHUNK_SIZE)
    channel = ScreenQr()
    manifest = Manifest("cat.png", len(PAYLOAD), CHUNK_SIZE, config.c, config.delta)
    return channel, chunks, config, manifest


def _droplet_image(channel, chunks, config, seed):
    droplet = encode_droplet(chunks, seed, config)
    frame_bytes = frame.build_frame(seed=droplet.seed, payload=droplet.payload)
    return channel.modulate(bytes_to_bits(frame_bytes))


def _manifest_image(channel, manifest):
    frame_bytes = frame.build_frame(seed=MANIFEST_SEED, payload=build_manifest(manifest))
    return channel.modulate(bytes_to_bits(frame_bytes))


def test_rate_meter_counts_only_recent_events():
    meter = RateMeter(window_s=2.0)

    meter.tick(0.0)  # 창 밖으로 밀려나야 하는 오래된 사건
    meter.tick(9.0)
    meter.tick(9.5)

    assert meter.rate(10.0) == 1.0  # 최근 2초 안의 2건 ÷ 2초


def test_blank_frame_is_counted_as_dropped():
    channel, *_ = _setup()
    session = QrReceiveSession(channel)

    note = session.feed(np.full((100, 100), 255, dtype=np.uint8))

    assert note is None
    assert session.frames_seen == 1
    assert session.frames_dropped == 1


def test_manifest_frame_starts_the_decoder():
    channel, _, _, manifest = _setup()
    session = QrReceiveSession(channel)

    note = session.feed(_manifest_image(channel, manifest))

    assert session.manifest == manifest
    assert session.decoder is not None
    assert note is not None and "cat.png" in note


def test_droplets_before_the_manifest_are_kept_and_replayed():
    """카메라를 늦게 들이대도 그 사이 받은 방울을 버리지 않는다."""
    channel, chunks, config, manifest = _setup()
    session = QrReceiveSession(channel)

    session.feed(_droplet_image(channel, chunks, config, seed=1))
    session.feed(_droplet_image(channel, chunks, config, seed=2))
    assert session.droplets_new == 2
    assert session.chunks_done == 0  # 아직 K를 모른다

    session.feed(_manifest_image(channel, manifest))

    assert session.decoder is not None
    assert session.decoder.droplets_received == 2  # 들고 있던 방울이 들어갔다


def test_same_seed_twice_counts_as_duplicate():
    channel, chunks, config, manifest = _setup()
    session = QrReceiveSession(channel, manifest)
    image = _droplet_image(channel, chunks, config, seed=1)

    session.feed(image)
    session.feed(image)

    assert session.droplets_new == 1
    assert session.droplets_dup == 1


def test_repeated_manifest_is_not_counted_as_a_new_droplet():
    channel, _, _, manifest = _setup()
    session = QrReceiveSession(channel)
    image = _manifest_image(channel, manifest)

    session.feed(image)
    session.feed(image)

    assert session.droplets_new == 0
    assert session.droplets_dup == 1


def test_a_file_sized_payload_is_recovered_byte_for_byte():
    """안내표 → 방울 → 복원까지, 화면과 카메라만 뺀 전체 흐름."""
    channel, chunks, config, manifest = _setup()
    session = QrReceiveSession(channel)
    session.feed(_manifest_image(channel, manifest))

    for seed in range(1, 400):
        if session.is_complete:
            break
        session.feed(_droplet_image(channel, chunks, config, seed))

    assert session.is_complete
    assert session.recovered_bytes() == PAYLOAD
    assert session.droplets_new >= len(chunks)  # 오버헤드 — 여기서 눈에 보인다


def test_frame_plan_leaves_room_for_fountain_overhead():
    """오버헤드가 1.4~2.0배이므로 계획은 그보다 여유가 있어야 한다."""
    planned = frame_plan(k=100, manifest_every=8)

    assert planned > 200


def test_goodput_is_zero_before_any_time_passes():
    assert goodput_kbps(1024, 0.0) == 0.0
    assert goodput_kbps(1024, 1.0) == 1.0
