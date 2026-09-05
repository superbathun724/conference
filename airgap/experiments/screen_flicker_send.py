"""화면 밝기 변조 채널 실측 시연 · 송신측.

M3 완료 조건("QR과 밝기 변조의 전송률 차이를 수치로 제시")을 확인하려면
밝기 변조 채널도 QR과 마찬가지로 실기기 측정이 있어야 한다. 이 스크립트는
음향 채널(acoustic_send.py)과 같은 구조를 그대로 따른다 — 밝기 변조는
QR과 달리 파운틴 방울을 흩뿌리는 방식이 아니라, 음향 FSK처럼 프레임 하나를
시간축으로 쭉 이어붙여 한 번에 보내는 채널이기 때문이다(둘 다
coding_axes=("time",)). 그래서 화면-카메라 짝을 시행마다 사람이 Enter로
맞추는 방식도 acoustic_send.py/acoustic_recv.py와 동일하다.

사용법 (두 대의 기기: 노트북 화면 = 송신, 다른 기기 카메라 = 수신):
    (수신측 먼저) python -m airgap.experiments.screen_flicker_recv --trials 10 --distance-cm 30
    (송신측)      python -m airgap.experiments.screen_flicker_send --trials 10

--out-dir을 주면 노트북 화면을 깜빡이는 대신 시행별 동영상 파일로 써낸다.
음향 채널의 --out-dir(WAV 저장)과 같은 성격의 경로다. 폰 A가 이 동영상을
전체화면으로 재생하고 폰 B가 녹화하면 노트북 화면 없이 안드로이드 두 대만으로
측정할 수 있다.

    python -m airgap.experiments.screen_flicker_send --out-dir out/flicker --trials 10
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from airgap.channels.screen_flicker import ScreenFlicker, ScreenFlickerConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.experiments import screen_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "channels" / "screen_flicker.yaml"
)
DEFAULT_MESSAGE = "AIRGAP 20자 테스트문자열"  # noqa: RUF001 (연구 목적의 한글 테스트 문자열)


def main() -> None:
    parser = argparse.ArgumentParser(description="화면 밝기 변조 송신 시연/실측")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="보낼 문자열")
    parser.add_argument("--trials", type=int, default=10, help="반복 횟수")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="지정하면 화면 깜빡임 대신 시행별 동영상을 이 폴더에 써낸다 (안드로이드 중계용)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="채널 설정 YAML 경로"
    )
    args = parser.parse_args()

    config = ScreenFlickerConfig.from_yaml(args.config)
    channel = ScreenFlicker(config)
    payload = args.message.encode("utf-8")

    log.info(
        "설정: message=%r trials=%d config=%s symbol_duration_ms=%s fps=%s",
        args.message,
        args.trials,
        args.config,
        config.symbol_duration_ms,
        config.fps,
    )

    if args.out_dir is not None:
        digits = max(2, len(str(args.trials)))

    for trial in range(1, args.trials + 1):
        frame_bytes = frame.build_frame(seed=trial, payload=payload)
        bits = bytes_to_bits(frame_bytes)
        signal = channel.modulate(bits)

        if args.out_dir is not None:
            # 밝기 값(0~1) 하나가 화면 한 장이다. 단색 정사각형으로 그린다.
            screens = [
                np.full((480, 480), round(float(level) * 255), dtype=np.uint8) for level in signal
            ]
            path = args.out_dir / f"trial_{trial:0{digits}d}.mp4"
            # 심볼 하나가 이미 여러 프레임으로 펼쳐져 나오므로 화면당 1프레임이다.
            written = screen_video.write_video(
                screens, path, display_ms=1000.0 / config.fps, video_fps=config.fps
            )
            log.info(
                "[%d/%d] 저장: %s (%d프레임, %.1f초)",
                trial,
                args.trials,
                path,
                written,
                written / config.fps,
            )
            continue

        input(
            f"[{trial}/{args.trials}] Enter를 누르면 화면을 깜빡입니다"
            " (수신측 촬영 시작 후 누르세요)..."
        )
        channel.emit(signal)
        log.info("[%d/%d] 표시 완료", trial, args.trials)

    if args.out_dir is not None:
        log.info(
            "완료: 동영상 %d개를 %s에 저장. 폰 A에서 전체화면·최대 밝기·자동회전 끄고"
            " 순서대로 재생하고, 폰 B로 녹화한 뒤 screen_flicker_recv.py --in-dir로 넘길 것"
            " (--expected-payload %r 도 함께 넘길 것)",
            args.trials,
            args.out_dir,
            args.message,
        )


if __name__ == "__main__":
    main()
