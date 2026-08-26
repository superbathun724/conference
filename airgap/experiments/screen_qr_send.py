"""화면 QR 채널 실측 시연 · 송신측.

M3 완료 조건("화면에서 카메라로 이미지 파일 1개 복원")을 실제 장치로
확인하기 위한 스크립트. screen_qr_recv.py와 짝을 이룬다.

음향 채널(acoustic_send.py)은 시행마다 프레임 하나를 보냈지만, 여기서는
파운틴 방울을 화면에 계속 바꿔가며 띄운다 — "매 프레임마다 QR 코드가
바뀌는" 시연이 이 방식이다. 수신측은 그중 몇 장을 놓치든 상관없이
필요한 만큼만 모이면 복원한다(core/fountain.py).

되돌아갈 통신로가 없는 오프라인 채널이므로 송신측은 수신측이 복원을
끝냈는지 알 방법이 없다. 그래서 --max-frames만큼 무조건 다 보여주고
끝난다 — 수신측이 그 전에 복원을 마쳤다면 남은 프레임은 안 봐도 됐던
것뿐이다.

사용법 (두 대의 기기: 노트북 화면 = 송신, 다른 기기 카메라 = 수신):
    (수신측 먼저) python -m airgap.experiments.screen_qr_recv --trials 5 --distance-cm 30
    (송신측)      python -m airgap.experiments.screen_qr_send --message "AIRGAP 20자 테스트문자열"
"""

import argparse
import logging
from pathlib import Path

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.core.fountain import LtConfig, encode_droplet, split_into_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNEL_CONFIG_PATH = _PROJECT_ROOT / "config" / "channels" / "screen_qr.yaml"
DEFAULT_FOUNTAIN_CONFIG_PATH = _PROJECT_ROOT / "config" / "fountain.yaml"
DEFAULT_MESSAGE = "AIRGAP 20자 테스트문자열"  # noqa: RUF001 (연구 목적의 한글 테스트 문자열)


def main() -> None:
    parser = argparse.ArgumentParser(description="화면 QR 송신 시연/실측 (파운틴 방울을 반복 표시)")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="보낼 문자열")
    parser.add_argument("--max-frames", type=int, default=60, help="화면에 띄울 최대 QR 프레임 수")
    parser.add_argument(
        "--channel-config", type=Path, default=DEFAULT_CHANNEL_CONFIG_PATH, help="QR 채널 설정 YAML"
    )
    parser.add_argument(
        "--fountain-config",
        type=Path,
        default=DEFAULT_FOUNTAIN_CONFIG_PATH,
        help="LT 부호 설정 YAML",
    )
    args = parser.parse_args()

    channel = ScreenQr(ScreenQrConfig.from_yaml(args.channel_config))
    fountain_config = LtConfig.from_yaml(args.fountain_config)
    payload = args.message.encode("utf-8")
    chunks = split_into_chunks(payload, fountain_config.chunk_size_bytes)
    k = len(chunks)

    log.info(
        "설정: message=%r k=%d chunk_size_bytes=%d max_frames=%d display_ms=%s",
        args.message,
        k,
        fountain_config.chunk_size_bytes,
        args.max_frames,
        channel.config.display_ms,
    )
    input("Enter를 누르면 QR 프레임을 순서대로 띄우기 시작합니다 (수신측을 먼저 실행해둘 것)...")

    try:
        for droplet_seed in range(args.max_frames):
            droplet = encode_droplet(chunks, droplet_seed, fountain_config)
            frame_bytes = frame.build_frame(seed=droplet.seed, payload=droplet.payload)
            bits = bytes_to_bits(frame_bytes)
            image = channel.modulate(bits)
            channel.emit(image)
            if (droplet_seed + 1) % 10 == 0:
                log.info("%d/%d 프레임 표시", droplet_seed + 1, args.max_frames)
    finally:
        # emit()은 창을 열어만 두고 안 닫는다(연속 표시 중 깜빡임 방지) — 여기서 한 번만 닫는다.
        channel.close_display()

    log.info("완료: %d개 프레임을 순서대로 표시했다 (k=%d)", args.max_frames, k)


if __name__ == "__main__":
    main()
