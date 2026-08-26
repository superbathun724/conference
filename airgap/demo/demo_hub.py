"""데모 화면이 공유하는 상태 저장소 (M5 Tier 1, `docs/DEMO_UI_PLAN.md` 참고).

**개념:** 발표용 데모는 "지금 TX인가 RX인가", "방금 무슨 일이 있었나(로그)" 같은
상태를 화면에 보여준다. 이 상태를 만드는 쪽(채널을 감싸는 코드)과 그리는 쪽
(터미널 패널, 나중에 만들 파형 창 등)을 분리해두면, 그리는 방식을 나중에
바꾸거나 늘려도(rich → matplotlib → 웹) 상태를 만드는 코드는 그대로 재사용된다.
`DemoHub`가 그 경계에 있는 "상태 저장소"다.

배경 스레드(마이크를 계속 듣는 스레드)와 메인 스레드(타이핑 받는 스레드)가
동시에 값을 갱신할 수 있으므로 `threading.Lock`으로 값 하나를 읽고 쓰는 동안
다른 스레드가 끼어들지 못하게 막는다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoStats:
    """한 순간의 데모 상태 스냅샷. `DemoHub.snapshot()`이 돌려주는 불변 값이다."""

    channel_name: str
    mode: str  # "TX" | "RX" | "IDLE"
    elapsed_s: float
    log_lines: tuple[str, ...]


class DemoHub:
    """여러 스레드가 안전하게 갱신·조회하는 데모 상태 저장소."""

    def __init__(self, channel_name: str, max_log_lines: int = 8) -> None:
        self._channel_name = channel_name
        self._max_log_lines = max_log_lines
        self._started_at = time.monotonic()
        self._lock = threading.Lock()
        self._mode = "IDLE"
        self._log_lines: list[str] = []

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def log(self, line: str) -> None:
        """로그 한 줄을 시각과 함께 남긴다. 오래된 줄은 max_log_lines를 넘으면 버린다."""
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._log_lines.append(f"{timestamp} {line}")
            if len(self._log_lines) > self._max_log_lines:
                self._log_lines = self._log_lines[-self._max_log_lines :]

    def snapshot(self) -> DemoStats:
        """지금 이 순간의 상태를 복사해서 돌려준다 (렌더러는 이 스냅샷만 본다)."""
        with self._lock:
            return DemoStats(
                channel_name=self._channel_name,
                mode=self._mode,
                elapsed_s=time.monotonic() - self._started_at,
                log_lines=tuple(self._log_lines),
            )
