"""demo/hud_rich.py의 render_panel() 테스트. 터미널 없이 rich Console(record=True)로 캡처한다."""

from rich.console import Console

from airgap.demo.demo_hub import DemoStats
from airgap.demo.hud_rich import render_panel


def _render_to_text(stats: DemoStats) -> str:
    console = Console(record=True, width=60)
    console.print(render_panel(stats))
    return console.export_text()


def test_panel_shows_channel_name_as_title():
    stats = DemoStats(channel_name="acoustic_fsk", mode="IDLE", elapsed_s=0.0, log_lines=())

    text = _render_to_text(stats)

    assert "ACOUSTIC_FSK" in text


def test_panel_shows_tx_mode():
    stats = DemoStats(channel_name="acoustic_fsk", mode="TX", elapsed_s=1.2, log_lines=())

    text = _render_to_text(stats)

    assert "TX" in text


def test_panel_shows_placeholder_when_no_log():
    stats = DemoStats(channel_name="acoustic_fsk", mode="IDLE", elapsed_s=0.0, log_lines=())

    text = _render_to_text(stats)

    assert "아직 기록 없음" in text


def test_panel_shows_log_lines():
    stats = DemoStats(
        channel_name="acoustic_fsk",
        mode="RX",
        elapsed_s=3.0,
        log_lines=("12:00:00 수신: hello world",),
    )

    text = _render_to_text(stats)

    assert "수신: hello world" in text
