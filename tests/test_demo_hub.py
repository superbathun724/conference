"""demo/demo_hub.py의 DemoHub 테스트. 스레드나 실제 채널 없이 상태 저장소만 검증한다."""

from airgap.demo.demo_hub import DemoHub


def test_snapshot_reflects_mode():
    hub = DemoHub(channel_name="acoustic_fsk")

    hub.set_mode("TX")

    assert hub.snapshot().mode == "TX"


def test_snapshot_channel_name_and_default_mode():
    hub = DemoHub(channel_name="acoustic_fsk")

    stats = hub.snapshot()

    assert stats.channel_name == "acoustic_fsk"
    assert stats.mode == "IDLE"


def test_log_lines_appear_in_snapshot():
    hub = DemoHub(channel_name="acoustic_fsk")

    hub.log("보냄: hello")

    assert len(hub.snapshot().log_lines) == 1
    assert "보냄: hello" in hub.snapshot().log_lines[0]


def test_log_truncates_to_max_lines():
    hub = DemoHub(channel_name="acoustic_fsk", max_log_lines=3)

    for i in range(5):
        hub.log(f"line {i}")

    lines = hub.snapshot().log_lines
    assert len(lines) == 3
    # 가장 최근 3개만 남아야 한다 (오래된 것부터 버림)
    assert "line 4" in lines[-1]
    assert "line 2" in lines[0]


def test_elapsed_s_is_non_negative():
    hub = DemoHub(channel_name="acoustic_fsk")

    assert hub.snapshot().elapsed_s >= 0.0


def test_update_sets_numeric_fields():
    hub = DemoHub(channel_name="screen_qr")

    hub.update(frames_seen=12, chunks_total=5)

    stats = hub.snapshot()
    assert stats.frames_seen == 12
    assert stats.chunks_total == 5


def test_update_rejects_unknown_field():
    hub = DemoHub(channel_name="screen_qr")

    try:
        hub.update(capure_fps=1.0)  # 오타
    except KeyError:
        pass
    else:
        raise AssertionError("오타 난 필드명은 KeyError로 걸러져야 한다")


def test_numeric_fields_default_to_zero_for_acoustic_demo():
    hub = DemoHub(channel_name="acoustic_fsk")

    stats = hub.snapshot()

    assert stats.chunks_total == 0
    assert stats.banner is None
