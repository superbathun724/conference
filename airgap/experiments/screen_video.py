"""화면 채널의 송신 화면을 동영상 파일로 써내는 도구 (안드로이드 중계용).

**왜 필요한가.** 음향 채널은 이미 스피커를 쓰지 않고 WAV 파일을 만들어
폰으로 옮겨 재생하는 경로가 있다(`acoustic_send.py --out-dir`). 노트북
오디오 드라이버가 고장났을 때 만든 우회로인데, 결과적으로 측정 원본이
파일로 남는 장점이 생겼다. 화면 채널에는 그에 해당하는 경로가 없어서
노트북 화면 없이는 실험을 할 수 없었다. 이 모듈이 그 빈자리를 메운다.

폰 A가 이 동영상을 전체화면으로 재생하고 폰 B가 그것을 녹화하면,
**노트북 화면 없이 안드로이드 두 대만으로** 가시광 채널을 측정할 수 있다.
판정은 세 채널 모두와 똑같이 노트북의 수신 스크립트가 맡으므로 판정
기준은 달라지지 않는다(CLAUDE.md 규칙 5).

**표시 시간을 프레임 수로 바꾼다.** 한 화면을 display_ms만큼 보여주려면
초당 몇 장짜리 동영상이냐에 따라 같은 화면을 여러 프레임 반복해 써야 한다.
동영상 자체를 2fps 같은 낮은 값으로 만들면 폰 재생기가 제대로 처리하지
못하는 경우가 있어, 파일은 항상 흔한 프레임률(기본 30fps)로 만들고 화면
하나를 그 프레임률에 맞는 횟수만큼 반복한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO_FPS = 30.0


def frames_per_screen(display_ms: float, video_fps: float = DEFAULT_VIDEO_FPS) -> int:
    """한 화면을 display_ms만큼 유지하려면 몇 프레임을 반복해 써야 하는가."""
    return max(1, round(display_ms / 1000.0 * video_fps))


def write_video(
    screens: list[np.ndarray],
    path: Path,
    display_ms: float,
    video_fps: float = DEFAULT_VIDEO_FPS,
    canvas_size: int | None = None,
) -> int:
    """화면(2차원 흑백 배열) 목록을 동영상 파일 하나로 써낸다.

    canvas_size를 주면 모든 화면을 그 크기의 정사각형 한가운데에 흰 여백과
    함께 놓는다. 화면마다 QR 크기가 달라도 동영상 프레임 크기는 하나로
    고정돼야 하고, 재생 중 그림 크기가 들쭉날쭉하면 카메라 초점이 계속
    다시 잡히기 때문이다.

    반환값은 실제로 쓴 프레임 수다.
    """
    if not screens:
        raise ValueError("써낼 화면이 없다")

    side = canvas_size or max(max(s.shape) for s in screens)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (side, side))
    if not writer.isOpened():
        raise RuntimeError(f"동영상 파일을 열 수 없다: {path}")

    repeat = frames_per_screen(display_ms, video_fps)
    written = 0
    try:
        for screen in screens:
            canvas = np.full((side, side), 255, dtype=np.uint8)
            top = (side - screen.shape[0]) // 2
            left = (side - screen.shape[1]) // 2
            canvas[top : top + screen.shape[0], left : left + screen.shape[1]] = screen
            bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            for _ in range(repeat):
                writer.write(bgr)
                written += 1
    finally:
        writer.release()
    return written
