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


def test_progress_bar_is_empty_when_total_is_zero():
    from airgap.demo.hud_rich import render_progress_bar

    assert render_progress_bar(0, 0) == ""


def test_progress_bar_fills_proportionally():
    from airgap.demo.hud_rich import render_progress_bar

    bar = render_progress_bar(5, 10, width=10)

    assert bar.count("█") == 5
    assert bar.count("░") == 5


def test_stats_table_hidden_for_acoustic_demo():
    """음향 채팅처럼 프레임·조각 숫자가 없는 데모에는 통계표를 그리지 않는다."""
    stats = DemoStats(channel_name="acoustic_fsk", mode="RX", elapsed_s=1.0, log_lines=())

    text = _render_to_text(stats)

    assert "CAPTURE FPS" not in text


def test_stats_table_shown_for_qr_demo():
    stats = DemoStats(
        channel_name="screen_qr",
        mode="RX",
        elapsed_s=2.0,
        log_lines=(),
        chunks_done=3,
        chunks_total=10,
        frames_seen=40,
    )

    text = _render_to_text(stats)

    assert "CAPTURE FPS" in text
    assert "3/10" in text


def test_banner_is_shown_when_transfer_completes():
    stats = DemoStats(
        channel_name="screen_qr",
        mode="IDLE",
        elapsed_s=4.0,
        log_lines=(),
        chunks_done=10,
        chunks_total=10,
        banner="전송 완료! — 160 B in 4.0s (0.04 KB/s)",
    )

    text = _render_to_text(stats)

    assert "전송 완료" in text


def test_level_bar_is_empty_for_silence():
    from airgap.demo.hud_rich import render_level_bar

    assert render_level_bar(0.0, width=10) == "░" * 10


def test_level_bar_grows_with_loudness():
    from airgap.demo.hud_rich import render_level_bar

    quiet = render_level_bar(0.001, width=20).count("█")
    loud = render_level_bar(0.5, width=20).count("█")

    assert quiet < loud


def test_info_lines_and_level_bar_are_shown():
    stats = DemoStats(
        channel_name="acoustic_fsk",
        mode="RX",
        elapsed_s=1.0,
        log_lines=(),
        info_lines=("채널   FSK 3.0/4.0 kHz",),
        level_bar="████░░░░",
    )

    text = _render_to_text(stats)

    assert "FSK 3.0/4.0 kHz" in text
    assert "████" in text
