"""QR 심볼로 비트를 화면에 표시하는 가시광 채널.

한 번의 modulate() 호출 = 프레임(core/frame.py) 하나 = QR 코드 이미지 한 장이다.
음향 채널이 심볼을 시간에 늘어놓듯, 이 채널은 조각을 시간(여러 프레임)과
공간(한 프레임 안의 2차원 격자) 두 축에 나눠 담는다. 그래서 같은 파운틴
부호·같은 프레임 포맷을 그대로 쓰면서도 화면 갱신률(fps)에 따라 전송률이
결정된다는 점에서 밝기 변조(screen_flicker.py)와 좋은 비교쌍이 된다 —
매질과 하드웨어는 같고 "공간 차원을 쓰는가"만 다르다.

QR 코드 자체의 생성·인식은 표준 부호(Reed-Solomon 기반)라 직접 구현하지
않고 라이브러리(qrcode, pyzbar)를 그대로 쓴다 (CLAUDE.md 규칙 2). 이 모듈이
직접 짜는 부분은 "우리 프레임 바이트열을 QR 페이로드에 얹고 다시 꺼내는"
연결부뿐이다.

프레임 바이트열은 base45(RFC 9285)로 감싸서 QR에 넣는다. 날바이트를 그대로
넣지 못하는 이유는 라이브러리 쪽 동작인데, 디코더(zbar)가 "8비트 바이트 모드"
QR 내용을 텍스트로 보고 임의로 문자 인코딩(예: Shift-JIS)을 추정해 UTF-8로
재인코딩해서 돌려준다 — 우리 프레임의 프리앰블·CRC 바이트(0x80 이상)가 이
과정에서 다른 바이트로 깨진다. ASCII 문자로 감싸면 이 오작동을 피해간다.

처음에는 base64를 썼지만 2026-09-03에 base45로 바꿨다. base45는 QR '영숫자
모드'에 들어가서 같은 조각을 더 작은 QR에 담는다(자세한 계산은
channels/base45.py 독스트링). 예전에 base64로 찍어둔 측정 원본을 다시 읽을 수
있어야 하므로(CLAUDE.md 규칙 3), 복조는 base45를 먼저 시도하고 실패하면
base64로도 시도한다.

base45로 적기 전에 프레임 바이트열을 한 번 뒤섞는다(_scramble). 파운틴 조각의
마지막 하나에 들어가는 0 채움 구간이 QR 생성기를 깨뜨리기 때문인데, 자세한
이유는 그 함수의 설명에 적었다.
"""

from __future__ import annotations

import base64
import binascii
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import qrcode
import yaml
from pyzbar.pyzbar import decode as zbar_decode
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q

from airgap.channels import base45
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.core.channel import Channel, ChannelCaps

_EC_LEVELS = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}


def _scramble(data: bytes) -> bytes:
    """프레임 바이트열에 고정된 난수열을 XOR해 같은 값이 길게 이어지지 않게 만든다.

    **왜 필요한가.** 파운틴 조각의 마지막 하나는 원본 길이를 맞추려고 0으로
    채워진다(core/fountain.py의 split_into_chunks). 파일 크기가 조각 크기의
    배수인 경우는 드무니 이건 예외가 아니라 거의 항상 생기는 일이다. 이 0
    구간을 base45로 적으면 '0' 문자만 길게 늘어서고, QR 영숫자 모드에서 그건
    값이 전부 0인 데이터가 된다. 그런데 qrcode 라이브러리는 오류정정 블록
    하나가 통째로 0이면 내부 계산에서 죽는다(glog(0)).

    **어떻게 막는가.** 보내기 전에 데이터에 정해진 난수열을 XOR해두면 0이
    이어지던 자리가 들쭉날쭉한 값으로 바뀐다. 실제 통신 규격(블루투스,
    지상파 디지털 방송 등)도 같은 이유로 같은 일을 하고, 이것을 스크램블링
    또는 화이트닝이라고 부른다. 난수열이 데이터와 무관하게 정해져 있으므로
    받는 쪽은 똑같은 XOR을 한 번 더 하면 원래대로 돌아온다 — 그래서 이 함수
    하나가 거는 쪽과 푸는 쪽을 겸한다.

    난수열은 아주 단순한 선형 합동 생성기(다음 값 = 이전 값 x 73 + 41)로
    만든다. 암호가 아니라 "값이 골고루 섞이기만 하면 되는" 용도라 이 정도면
    충분하다.
    """
    out = bytearray(len(data))
    state = 0x5A  # 시작값. 0만 아니면 무엇이든 된다 — 송수신이 같기만 하면 된다
    for i, byte in enumerate(data):
        state = (state * 73 + 41) & 0xFF
        out[i] = byte ^ state
    return bytes(out)


@dataclass(frozen=True)
class ScreenQrConfig:
    """QR 채널의 파라미터. config/channels/screen_qr.yaml에서 읽는다."""

    box_size: int = 8  # QR 모듈(정사각형 점) 하나의 픽셀 크기
    border: int = 4  # 여백 모듈 수 (조용한 구역, quiet zone) — 너무 좁으면 인식률이 떨어진다
    error_correction: str = "M"  # L(7%)/M(15%)/Q(25%)/H(30%) — 얼룩·초점 흐림에 대한 여유
    display_ms: float = 500.0  # 프레임 하나를 화면에 띄워두는 시간
    fullscreen: bool = True
    camera_index: int | None = None
    capture_timeout_s: float = 2.0  # capture()가 프레임을 못 읽으면 포기하는 시간

    @classmethod
    def from_yaml(cls, path: Path) -> ScreenQrConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ScreenQr(Channel):
    """QR 코드 이미지 한 장 = 프레임 하나로 대응시키는 가시광 채널."""

    _WINDOW_NAME = "airgap_screen_qr"

    def __init__(self, config: ScreenQrConfig | None = None) -> None:
        self.config = config or ScreenQrConfig()
        self._ec_level = _EC_LEVELS[self.config.error_correction]
        self._window_open = False
        frames_per_sec = 1000.0 / self.config.display_ms
        self.caps = ChannelCaps(
            name="screen_qr",
            medium="em_visible",
            wave_type="electromagnetic",
            coding_axes=("time", "space"),
            directional=True,
            penetrates_opaque=False,
            human_perceptible=True,
            nominal_bps=frames_per_sec,  # 프레임 하나에 프레임(가변 바이트) 하나 — 참고용 기준치
        )

    # ---- 변조/복조: 장치 없이 이미지 배열만 다룬다 (루프백 가능) ----

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """비트 배열 → QR 코드 흑백 이미지(2차원 배열, 0~255)."""
        payload = base45.encode(_scramble(bits_to_bytes(bits)))
        qr = qrcode.QRCode(
            error_correction=self._ec_level,
            box_size=self.config.box_size,
            border=self.config.border,
        )
        qr.add_data(payload)
        qr.make(fit=True)
        image = qr.make_image(fill_color="black", back_color="white").convert("L")
        return np.array(image, dtype=np.uint8)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """QR 이미지 → 비트 배열. 2차원이 아니거나 QR을 못 찾으면 빈 배열.

        `Channel` 인터페이스는 신호 하나에 프레임 하나를 돌려주므로, 화면에
        QR이 여러 장 있어도 첫 장만 준다. 전부 필요하면 demodulate_all()을 쓴다.
        """
        frames = self.demodulate_all(signal)
        if not frames:
            return np.array([], dtype=np.uint8)
        return frames[0]

    def demodulate_all(self, signal: np.ndarray) -> list[np.ndarray]:
        """한 장의 이미지에서 찾은 QR을 **전부** 비트 배열로 돌려준다.

        화면에 QR을 격자로 여러 개 띄우는 실험(experiments/qr_grid.py)에서
        쓴다. 이미지를 격자 모양대로 잘라 나눌 필요는 없다 — zbar가 한 이미지
        안의 심볼을 알아서 다 찾아주기 때문이다. 잘라 나누는 방식은 화면이
        카메라 화각의 어디에 어떻게 걸렸는지에 의존하지만, 이 방식은 그런
        가정이 없어서 실기기에서 훨씬 튼튼하다.
        """
        if signal.ndim != 2 or signal.size == 0:
            return []

        out: list[np.ndarray] = []
        for result in zbar_decode(signal.astype(np.uint8)):
            payload = self._decode_symbol(result.data)
            if payload is not None:
                out.append(bytes_to_bits(payload))
        return out

    @staticmethod
    def _decode_symbol(data: bytes) -> bytes | None:
        """QR 심볼 내용 → 프레임 바이트열. 어느 방식으로도 못 풀면 None.

        base45를 먼저 시도하고 base64로 물러선다. base64는 2026-09-03 이전에
        찍어둔 측정 원본(data/raw의 동영상)을 지금 코드로 다시 판정할 수 있게
        남겨둔 경로다 — 새로 보내는 쪽은 base45만 쓴다.
        """
        try:
            # 스크램블은 XOR이라 같은 함수를 한 번 더 걸면 풀린다.
            return _scramble(base45.decode(data.decode("ascii")))
        except (ValueError, UnicodeDecodeError):
            pass
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            return None

    # ---- 실제 장치 ----

    def emit(self, signal: np.ndarray) -> None:
        """QR 이미지를 화면에 표시한다 (카메라가 있는 다른 기기가 촬영).

        연속된 여러 프레임(파운틴 방울)을 보여줄 때 매번 창을 새로 열고 닫으면
        화면이 껐다 켰다 하는 것처럼 깜빡인다. 그래서 창은 첫 emit()에서 한 번만
        열고, 이후로는 내용만 갈아끼운다(imshow) — 창을 실제로 닫는 건
        close_display()가 맡는다. 시행이 끝나면 반드시 close_display()를 호출해야
        다음 시행이나 다른 채널이 같은 이름의 창을 새로 열 때 꼬이지 않는다.
        """
        if not self._window_open:
            flag = cv2.WINDOW_NORMAL if not self.config.fullscreen else cv2.WND_PROP_FULLSCREEN
            cv2.namedWindow(self._WINDOW_NAME, flag)
            if self.config.fullscreen:
                cv2.setWindowProperty(
                    self._WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
            self._window_open = True
        cv2.imshow(self._WINDOW_NAME, signal)
        cv2.waitKey(max(1, round(self.config.display_ms)))

    def close_display(self) -> None:
        """emit()이 열어둔 창을 닫는다. 한 시행(메시지 전체)이 끝난 뒤 한 번 호출한다."""
        if self._window_open:
            cv2.destroyWindow(self._WINDOW_NAME)
            self._window_open = False

    def capture(self, duration_s: float) -> np.ndarray:
        """카메라에서 프레임을 읽어 마지막으로 잡힌 한 장을 돌려준다.

        여러 장 읽고 마지막 것만 쓰는 이유: 노출·초점이 안정되기까지
        몇 프레임이 걸리는 카메라가 많아, 첫 프레임을 바로 쓰면 흐리게
        찍히는 경우가 있다.
        """
        index = self.config.camera_index if self.config.camera_index is not None else 0
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없다 (camera_index={index})")

        try:
            deadline = time.monotonic() + min(duration_s, self.config.capture_timeout_s)
            last_gray: np.ndarray | None = None
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                if ok:
                    last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_gray is None:
                raise RuntimeError("카메라에서 프레임을 하나도 읽지 못했다")
            return last_gray
        finally:
            cap.release()
