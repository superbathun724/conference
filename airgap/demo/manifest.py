"""전송할 내용을 설명하는 한 줄짜리 안내표(manifest)와 그 해석.

**왜 필요한가.** 문자열 하나를 보낼 때는 수신측에 `--message`로 같은 원문을
알려주고 거기서 조각 수 K를 계산하면 됐다. 파일을 보낼 때 그렇게 하면
"파일을 받으려면 그 파일을 이미 갖고 있어야 한다"는 우스운 상황이 된다.
그래서 파일 이름·크기·조각 설정을 담은 프레임을 따로 만들어 주기적으로
같이 내보낸다 — 수신측은 이걸 먼저 잡아야 복원을 시작할 수 있다.

**되돌아갈 통신로가 없다는 제약이 설계를 정한다.** 수신측이 "안내표 주세요"라고
요청할 방법이 없으므로, 송신측이 몇 프레임마다 한 번씩 계속 다시 내보낸다.
늦게 카메라를 들이대도 곧 하나를 잡게 된다. 파운틴 부호가 재전송 요청 없이
동작하는 것과 같은 발상이다.

**시드 0을 안내표 전용으로 쓴다.** 프레임에는 이미 16비트 시드 칸이 있으므로
새 필드를 만들 필요가 없다 — 방울 시드를 1부터 매기고 0을 비워두면, 수신측은
시드만 보고 "이건 안내표"를 구분할 수 있다. `core/frame.py`는 한 글자도 바꾸지
않는다(모든 채널이 공유하는 포맷이라 건드리면 기존 측정과 비교가 끊긴다).

형식은 사람이 눈으로 읽을 수 있는 한 줄이다. 발표장에서 QR 하나를 폰으로
찍어 보면 그대로 글자가 나오므로, "이 안에 뭐가 들었나"를 설명하기 쉽다:

    AIRGAP1|cat.png|10240|192|0.1|0.05
    (규격  |파일명 |바이트|조각|c  |delta)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

MAGIC = "AIRGAP1"
SEPARATOR = "|"
MANIFEST_SEED = 0  # 방울 시드는 1부터 — 0은 안내표 전용으로 비워둔다
MAX_NAME_BYTES = 64
_FIELD_COUNT = 6


@dataclass(frozen=True)
class Manifest:
    """전송 중인 내용에 대한 설명. 수신측은 이걸 받아야 디코더를 만들 수 있다."""

    name: str
    size_bytes: int
    chunk_size_bytes: int
    c: float
    delta: float

    @property
    def k(self) -> int:
        """원본 조각 수. `fountain.split_into_chunks`가 나누는 개수와 반드시 같아야 한다."""
        return max(1, math.ceil(self.size_bytes / self.chunk_size_bytes))


def safe_name(raw: str) -> str:
    """파일명에서 경로와 구분자를 걷어낸다.

    받은 이름을 그대로 저장 경로에 쓰면 `../../중요한파일` 같은 값으로 엉뚱한 곳에
    쓸 수 있다. 디렉터리 부분을 버리고 파일명만 남긴다. 구분자로 쓰는 `|`와 줄바꿈은
    안내표 자체를 깨뜨리므로 밑줄로 바꾼다.
    """
    name = Path(raw).name
    for bad in (SEPARATOR, "\n", "\r", "\t"):
        name = name.replace(bad, "_")
    name = name.strip() or "payload.bin"
    while len(name.encode("utf-8")) > MAX_NAME_BYTES:
        name = name[:-1]
    return name


def build_manifest(manifest: Manifest) -> bytes:
    """Manifest → 프레임에 실을 바이트열."""
    line = SEPARATOR.join(
        [
            MAGIC,
            safe_name(manifest.name),
            str(manifest.size_bytes),
            str(manifest.chunk_size_bytes),
            f"{manifest.c:g}",
            f"{manifest.delta:g}",
        ]
    )
    return line.encode("utf-8")


def parse_manifest(payload: bytes) -> Manifest | None:
    """바이트열 → Manifest. 우리 안내표가 아니거나 형식이 깨졌으면 None.

    카메라가 잡은 프레임 중에는 절반만 찍힌 것도 있으므로, 이상하면 조용히
    None을 돌려주고 다음 프레임을 기다리는 편이 낫다 — 예외를 던지면 데모가
    화면 하나 잘못 찍힌 것 때문에 통째로 멈춘다.
    """
    try:
        line = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None

    parts = line.split(SEPARATOR)
    if len(parts) != _FIELD_COUNT or parts[0] != MAGIC:
        return None

    try:
        size_bytes = int(parts[2])
        chunk_size_bytes = int(parts[3])
        c = float(parts[4])
        delta = float(parts[5])
    except ValueError:
        return None

    if size_bytes < 0 or chunk_size_bytes <= 0:
        return None

    return Manifest(
        name=safe_name(parts[1]),
        size_bytes=size_bytes,
        chunk_size_bytes=chunk_size_bytes,
        c=c,
        delta=delta,
    )
