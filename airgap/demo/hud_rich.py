"""`DemoStats`를 rich 패널로 그리는 Tier 1 터미널 HUD (`docs/DEMO_UI_PLAN.md` 참고).

**개념 (rich):** `rich`는 터미널에 색깔·테두리·표를 그릴 수 있게 해주는 라이브러리다.
여기서 쓰는 조각은 두 가지뿐이다 — `Text`(색 있는 글자 한 줄)와 `Panel`(그 글자들을
테두리로 감싸는 상자). 화면 전체를 실시간으로 새로 그리는 `Live` 기능은 입력
프롬프트(`input()`)와 화면을 다시 그리는 시점이 겹치면 꼬이기 쉬워서, 이번 Tier 1
범위에서는 쓰지 않는다 — 대신 "상태가 바뀔 때마다 패널을 한 번 새로 출력"하는
방식으로 단순하게 간다.

이 모듈은 화면에 아무것도 직접 출력하지 않는다 — `render_panel()`은 순수 함수라
`Panel` 객체만 돌려주고, 실제로 찍는 건 호출한 쪽(`console.print(...)`)이 한다.
그래야 터미널 없이도(pytest에서) 패널 내용을 테스트할 수 있다.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from airgap.demo.demo_hub import DemoStats

_MODE_STYLE = {
    "TX": ("● TX", "bold yellow"),
    "RX": ("● RX", "bold cyan"),
    "IDLE": ("○ IDLE", "grey50"),
}


def render_panel(stats: DemoStats) -> Panel:
    """DemoStats 한 장 → rich Panel. 순수 함수라 터미널 없이도 테스트 가능."""
    mode_label, mode_style = _MODE_STYLE.get(stats.mode, (stats.mode, "white"))

    header = Text()
    header.append(mode_label, style=mode_style)
    header.append(f"   경과 {stats.elapsed_s:5.1f}s", style="grey50")

    log_text = Text("\n".join(stats.log_lines) if stats.log_lines else "(아직 기록 없음)")

    return Panel(
        Group(header, Text(""), log_text),
        title=stats.channel_name.upper(),
        border_style="green",
    )
