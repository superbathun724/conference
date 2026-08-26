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
