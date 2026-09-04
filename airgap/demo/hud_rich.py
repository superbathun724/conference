"""`DemoStats`를 rich 패널로 그리는 Tier 1 터미널 HUD (`docs/DEMO_UI_PLAN.md` 참고).

**개념 (rich):** `rich`는 터미널에 색깔·테두리·표를 그릴 수 있게 해주는 라이브러리다.
여기서 쓰는 조각은 세 가지뿐이다 — `Text`(색 있는 글자 한 줄), `Table`(칸이 맞춰진 표),
`Panel`(그것들을 테두리로 감싸는 상자). 화면 전체를 실시간으로 새로 그리는 `Live`
기능은 입력 프롬프트(`input()`)와 화면을 다시 그리는 시점이 겹치면 꼬이기 쉬워서,
이번 Tier 1 범위에서는 쓰지 않는다 — 대신 "상태가 바뀔 때마다 패널을 한 번 새로
출력"하는 방식으로 단순하게 간다.

이 모듈은 화면에 아무것도 직접 출력하지 않는다 — `render_panel()`은 순수 함수라
`Panel` 객체만 돌려주고, 실제로 찍는 건 호출한 쪽(`console.print(...)`)이 한다.
그래야 터미널 없이도(pytest에서) 패널 내용을 테스트할 수 있다.

**통계표에 대하여 (2026-09-02 추가):** 화면 QR 데모는 프레임 수·유실·복원 진행도를
같이 보여준다. 음향 채팅에는 그 숫자가 아예 없으므로, 프레임도 조각도 0인 데모에는
통계표를 그리지 않는다 — 한 함수가 두 데모를 다 담당하되 없는 값을 0으로 늘어놓지는
않게 하려는 것이다. 진행 막대는 복원할 조각 수(K)를 아는 수신측에서만 그린다.
"""

from __future__ import annotations

import math

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from airgap.demo.demo_hub import DemoStats

_MODE_STYLE = {
    "TX": ("● TX", "bold yellow"),
    "RX": ("● RX", "bold cyan"),
    "IDLE": ("○ IDLE", "grey50"),
}

_BAR_WIDTH = 24  # 진행 막대의 칸 수


def render_progress_bar(done: int, total: int, width: int = _BAR_WIDTH) -> str:
    """확정 조각 수를 막대 문자열로 만든다 (순수 함수라 따로 테스트하기 쉽다).

    파운틴 부호는 "몇 %가 왔는지"가 아니라 "K개 중 몇 개가 확정됐는지"가 진행도다.
    받은 방울 수로 재면 100%를 넘어가 버리는데(오버헤드 1.8배 전후, FINDINGS.md
    2026-08-20 참고), 그건 진행도가 아니라 낭비된 양이므로 분모로 쓰지 않는다.
    """
    if total <= 0:
        return ""
    filled = min(width, round(width * done / total))
    return "█" * filled + "░" * (width - filled)


def render_level_bar(rms: float, width: int = _BAR_WIDTH) -> str:
    """소리의 실효값(RMS)을 막대로 바꾼다. SOUNDCHAT의 파형 바에 해당하는 Tier 1 표현.

    사람의 소리 감각도, 마이크의 동적 범위도 배수(로그)로 움직이기 때문에 진폭을
    그대로 쓰면 막대가 거의 항상 바닥에 붙어 있다. 그래서 dBFS(가장 큰 값을 0으로
    두고 로그를 취한 값)로 바꾼 뒤 -60dB~0dB 구간을 막대 전체 길이에 대응시킨다.
    """
    if rms <= 0.0:
        return "░" * width
    dbfs = 20.0 * math.log10(min(rms, 1.0))
    ratio = max(0.0, min(1.0, (dbfs + 60.0) / 60.0))
    filled = round(width * ratio)
    return "█" * filled + "░" * (width - filled)


def _stats_table(stats: DemoStats) -> Table:
    """DECIMEN류 통계 패널에 해당하는 2열 표. 값 계산은 하지 않고 받은 값을 늘어놓기만 한다."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="grey50", justify="right")
    table.add_column(style="bold")
    table.add_column(style="grey50", justify="right")
    table.add_column(style="bold")

    table.add_row(
        "CAPTURE FPS",
        f"{stats.capture_fps:.1f}",
        "DECODE FPS",
        f"{stats.decode_fps:.1f}",
    )
    table.add_row(
        "FRAMES",
        f"{stats.frames_seen}",
        "DROPPED",
        f"{stats.frames_dropped}",
    )
    table.add_row(
        "NEW/DUP/RED",
        f"{stats.droplets_new}/{stats.droplets_dup}/{stats.droplets_redundant}",
        "GOODPUT",
        f"{stats.goodput_kbps:.2f} KB/s",
    )
    chunks = f"{stats.chunks_done}/{stats.chunks_total}" if stats.chunks_total > 0 else "—"
    table.add_row(
        "PAYLOAD",
        f"{stats.payload_bytes} B",
        "CHUNKS",
        chunks,
    )
    return table


def render_panel(stats: DemoStats) -> Panel:
    """DemoStats 한 장 → rich Panel. 순수 함수라 터미널 없이도 테스트 가능."""
    mode_label, mode_style = _MODE_STYLE.get(stats.mode, (stats.mode, "white"))

    header = Text()
    header.append(mode_label, style=mode_style)
    header.append(f"   경과 {stats.elapsed_s:5.1f}s", style="grey50")

    parts: list[object] = [header, Text("")]

    if stats.chunks_total > 0 or stats.frames_seen > 0:
        parts.append(_stats_table(stats))
        if stats.chunks_total > 0:
            bar = render_progress_bar(stats.chunks_done, stats.chunks_total)
            parts.append(Text(f"  {bar}  {stats.chunks_done}/{stats.chunks_total}", style="cyan"))
        parts.append(Text(""))

    for line in stats.info_lines:
        parts.append(Text(line, style="grey62"))
    if stats.info_lines:
        parts.append(Text(""))

    if stats.level_bar:
        parts.append(Text(f"  {stats.level_bar}", style="cyan"))
        parts.append(Text(""))

    if stats.banner:
        parts.append(Text(f"✔ {stats.banner}", style="bold green"))
        parts.append(Text(""))

    parts.append(Text("\n".join(stats.log_lines) if stats.log_lines else "(아직 기록 없음)"))

    return Panel(
        Group(*parts),
        title=stats.channel_name.upper(),
        border_style="green" if stats.banner else "cyan",
    )
