"""데모 화면이 공유하는 상태 저장소 (M5 Tier 1, `docs/DEMO_UI_PLAN.md` 참고).

**개념:** 발표용 데모는 "지금 TX인가 RX인가", "방금 무슨 일이 있었나(로그)" 같은
상태를 화면에 보여준다. 이 상태를 만드는 쪽(채널을 감싸는 코드)과 그리는 쪽
(터미널 패널, 나중에 만들 파형 창 등)을 분리해두면, 그리는 방식을 나중에
바꾸거나 늘려도(rich → matplotlib → 웹) 상태를 만드는 코드는 그대로 재사용된다.
`DemoHub`가 그 경계에 있는 "상태 저장소"다.

배경 스레드(마이크를 계속 듣는 스레드)와 메인 스레드(타이핑 받는 스레드)가
동시에 값을 갱신할 수 있으므로 `threading.Lock`으로 값 하나를 읽고 쓰는 동안
다른 스레드가 끼어들지 못하게 막는다.

**필드가 두 묶음인 이유 (2026-09-02 추가):** 화면 QR 데모(`screen_qr_demo.py`)는
음향 채팅과 달리 초당 프레임 수·유실 프레임 수·복원 진행도 같은 숫자를 계속
보여준다. 이 숫자들은 `update()`로 넣고, 음향 채팅처럼 숫자가 필요 없는 데모는
그냥 안 넣으면 된다 — 기본값이 있어서 `DemoStats`는 그대로 만들어진다. 필드를
채널별로 나누지 않고 하나로 합쳐 둔 이유는, 렌더러가 데모 종류마다 다른 스냅샷
타입을 알아야 하는 상황을 피하기 위해서다.

음향 데모는 QR과 보여줄 것이 다르다 — 프레임 수나 복원 진행도가 아니라 반송
주파수·심볼 길이 같은 채널 설정과 지금 들리는 소리 크기다. QR의 고정된 표에
억지로 끼워 넣으면 "FRAMES"가 음향에서 무슨 뜻인지 아무도 답할 수 없게 되므로,
자유 형식 줄(`info_lines`)과 레벨 막대(`level_bar`)를 따로 뒀다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DemoStats:
    """한 순간의 데모 상태 스냅샷. `DemoHub.snapshot()`이 돌려주는 불변 값이다."""

    channel_name: str
    mode: str  # "TX" | "RX" | "IDLE"
    elapsed_s: float
    log_lines: tuple[str, ...]

    # --- 아래는 화면 QR 데모용 숫자. 안 쓰는 데모는 기본값 그대로 둔다. ---
    capture_fps: float = 0.0  # 카메라에서 초당 몇 장을 읽었나
    decode_fps: float = 0.0  # 그중 초당 몇 장에서 QR을 실제로 읽어냈나
    frames_seen: int = 0  # 지금까지 처리한 카메라 프레임 수
    frames_dropped: int = 0  # QR을 못 찾았거나 프레임 검사(CRC)에 걸린 프레임 수
    droplets_new: int = 0  # 처음 보는 시드의 방울
    droplets_dup: int = 0  # 이미 본 시드 (같은 QR을 연속으로 여러 장 찍으면 늘어난다)
    droplets_redundant: int = 0  # 새 방울인데 확정 조각 수를 당장 늘리지 못한 것
    chunks_done: int = 0  # 확정된 원본 조각 수
    chunks_total: int = 0  # 원본 조각 수 K (0이면 "전송 진행도가 없는 데모"로 본다)
    goodput_kbps: float = 0.0  # 확정된 조각 기준 실효 전송률
    payload_bytes: int = 0  # 보내려는 원문 바이트 수
    banner: str | None = None  # 완료 배너 (예: "전송 완료! ...")

    # --- 아래는 음향 데모용. QR처럼 고정된 표가 아니라 자유 형식이 필요해서 따로 둔다. ---
    info_lines: tuple[str, ...] = ()  # "라벨  값" 형태의 줄들 (채널 설정, 수신 현황 등)
    level_bar: str = ""  # 지금 들어오는 소리 크기를 나타내는 막대


# update()가 받을 수 있는 키 = DemoStats 필드에서 아래 4개(항상 직접 관리)를 뺀 나머지.
_MANAGED_FIELDS = {"channel_name", "mode", "elapsed_s", "log_lines"}
_UPDATABLE_FIELDS = {f.name for f in fields(DemoStats)} - _MANAGED_FIELDS


class DemoHub:
    """여러 스레드가 안전하게 갱신·조회하는 데모 상태 저장소."""

    def __init__(self, channel_name: str, max_log_lines: int = 8) -> None:
        self._channel_name = channel_name
        self._max_log_lines = max_log_lines
        self._started_at = time.monotonic()
        self._lock = threading.Lock()
        self._mode = "IDLE"
        self._log_lines: list[str] = []
        self._values: dict[str, object] = {}

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def update(self, **values: object) -> None:
        """숫자 상태를 갱신한다. 키 이름은 `DemoStats`의 필드명과 똑같이 쓴다.

        키를 먼저 검사해서 오타를 그 자리에서 잡는다 — 안 그러면 `capture_fps`를
        `capure_fps`로 잘못 써도 조용히 무시돼서, 발표장에서 "왜 숫자가 0이지"를
        디버깅하게 된다.
        """
        unknown = set(values) - _UPDATABLE_FIELDS
        if unknown:
            raise KeyError(f"DemoStats에 없는 필드: {sorted(unknown)}")
        with self._lock:
            self._values.update(values)

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
                **self._values,  # type: ignore[arg-type]
            )
