"""QR을 한 화면에 여러 장 띄우는 격자 배치 (축 A 실험 변인).

**무엇을 재려는 것인가.** 이 프로젝트의 축 A는 "정보를 얼마나 빠르게 보낼 수
있는가"이고, QR 채널이 밝기 변조(screen_flicker)와 다른 점은 오직 하나 —
공간 축을 쓴다는 것이다(README의 채널 표에서 coding_axes가 "시간+공간"). 화면을
n x n으로 나눠 QR을 동시에 n^2장 띄우면 **같은 하드웨어·같은 화면 갱신 속도로
공간 축만 더 쓰는 것**이 되므로, 전송률이 실제로 몇 배가 되는지를 통제된
조건에서 잴 수 있다. 이 모듈은 그 격자 수 n을 실험 변인으로 만든다.

**격자는 기본값이 아니다.** grid=1(한 화면에 QR 한 장)이 대조군이고, 밝기
변조와의 비교(M3 완료 조건)는 반드시 grid=1로 한다. 격자를 기본으로 켜면
채널 간 비교가 "공간 축을 쓰느냐"가 아니라 "QR만 최적화했느냐"가 되어
CLAUDE.md 규칙 5(채널 간 공정 비교)에 어긋난다.

**공짜가 아니다.** n을 올리면 QR 한 장이 화면에서 차지하는 크기가 1/n로
줄어든다(변 기준). 모듈(QR의 점) 하나가 카메라 화소 몇 개에 찍히느냐가
인식률을 좌우하므로, 어느 n부터 인식이 무너지는지가 바로 이 실험이 재려는
한계다. "n^2배 빨라진다"가 아니라 "몇 배까지 가다가 어디서 꺾이는가"를
재는 것이다.

**수신측은 이 모듈을 쓰지 않는다.** 이미지를 격자대로 잘라 나눌 필요가 없기
때문이다 — zbar가 한 이미지 안의 QR을 전부 찾아주므로 수신측은
ScreenQr.demodulate_all()만 부르면 된다. 잘라 나누는 방식은 화면이 카메라
화각에 어떻게 걸렸는지에 의존하지만, 이 방식은 그런 가정이 없다.
"""

from __future__ import annotations

import numpy as np

WHITE = 255


def tile(images: list[np.ndarray], grid: int) -> np.ndarray:
    """QR 이미지 여러 장을 grid x grid로 붙여 한 장으로 만든다.

    이미지 수가 grid*grid보다 적으면 남는 칸은 흰색으로 둔다(그 칸에는 QR이
    없는 것뿐이라 수신측이 알아서 무시한다). 칸 크기는 가장 큰 이미지에
    맞추고, 작은 이미지는 칸 왼쪽 위에 놓은 뒤 나머지를 흰색으로 채운다 —
    QR 주변의 흰 여백(조용한 구역)은 넓을수록 인식에 유리하므로 이렇게 채워도
    문제되지 않는다.
    """
    if grid < 1:
        raise ValueError(f"격자 수는 1 이상이어야 한다: {grid}")
    if not images:
        raise ValueError("붙일 이미지가 없다")
    if len(images) > grid * grid:
        raise ValueError(f"이미지가 칸 수보다 많다: {len(images)}장 > {grid * grid}칸")

    cell_h = max(img.shape[0] for img in images)
    cell_w = max(img.shape[1] for img in images)

    canvas = np.full((cell_h * grid, cell_w * grid), WHITE, dtype=np.uint8)
    for index, img in enumerate(images):
        row, col = divmod(index, grid)
        top, left = row * cell_h, col * cell_w
        canvas[top : top + img.shape[0], left : left + img.shape[1]] = img
    return canvas


def droplets_per_screen(grid: int) -> int:
    """한 화면에 실리는 방울 수. 이름을 붙여두면 계산식이 읽기 쉬워진다."""
    return grid * grid
