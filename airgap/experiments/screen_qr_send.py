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

--grid로 한 화면에 QR을 여러 장(grid x grid) 동시에 띄울 수 있다. 기본값 1이
대조군이고, 2 이상은 축 A의 "공간 축을 더 쓰면 전송률이 오르는가"를 재는
실험 조건이다(airgap/experiments/qr_grid.py 참고). **밝기 변조 채널과 비교하는
측정(M3 완료 조건)은 반드시 --grid 1로 한다** — 그래야 두 채널의 조건이 같다.

사용법 (두 대의 기기: 노트북 화면 = 송신, 다른 기기 카메라 = 수신):
    (수신측 먼저) python -m airgap.experiments.screen_qr_recv --trials 5 --distance-cm 30
    (송신측)      python -m airgap.experiments.screen_qr_send --message "AIRGAP 20자 테스트문자열"

    # 2x2 격자 조건 (수신측도 --grid 2로 맞춰 기록해야 한다)
    python -m airgap.experiments.screen_qr_send --message "..." --grid 2

--out-dir을 주면 노트북 화면에 띄우는 대신 동영상 파일로 써낸다. 음향 채널의
--out-dir(WAV 저장)과 같은 성격의 경로다. 폰 A가 이 동영상을 전체화면으로
재생하고 폰 B가 녹화하면 **노트북 화면 없이 안드로이드 두 대만으로** 측정할
수 있다. 판정은 어느 경로든 screen_qr_recv.py가 똑같이 맡는다.

    python -m airgap.experiments.screen_qr_send --out-dir out/qr --trials 5
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.core.fountain import LtConfig, encode_droplet, split_into_chunks
from airgap.experiments import qr_grid, screen_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNEL_CONFIG_PATH = _PROJECT_ROOT / "config" / "channels" / "screen_qr.yaml"
DEFAULT_FOUNTAIN_CONFIG_PATH = _PROJECT_ROOT / "config" / "fountain.yaml"
DEFAULT_MESSAGE = "AIRGAP 20자 테스트문자열"  # noqa: RUF001 (연구 목적의 한글 테스트 문자열)


def _load_payload(args) -> tuple[bytes, str]:
    """--file이 있으면 파일 내용을, 없으면 --message를 페이로드로 쓴다."""
    if args.file is not None:
        data = args.file.read_bytes()
        return data, f"{args.file.name} ({len(data)} B)"
    data = args.message.encode("utf-8")
    return data, f"message={args.message!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="화면 QR 송신 시연/실측 (파운틴 방울을 반복 표시)")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="보낼 문자열")
    parser.add_argument("--max-frames", type=int, default=60, help="화면에 띄울 최대 QR 프레임 수")
    parser.add_argument(
<<<<<<< HEAD
        "--chunk-size",
        type=int,
        default=None,
        help="파운틴 조각 크기(B). 기본은 config/fountain.yaml. 최대 255. 송수신이 같아야 한다",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="--message 대신 이 파일의 내용을 보낸다 (이미지 등 큰 페이로드). 수신측도 같은 --file",
    )
    parser.add_argument(
=======
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
        "--grid",
        type=int,
        default=1,
        help="한 화면에 띄울 QR 격자 수(n이면 n x n장). 1이 대조군, 2 이상은 축 A 실험 조건",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="지정하면 화면 표시 대신 시행별 동영상을 이 폴더에 써낸다 (안드로이드 중계용)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="--out-dir로 만들 동영상 개수. 화면 표시 모드에서는 무시된다",
    )
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
    if args.grid < 1:
        raise SystemExit(f"--grid는 1 이상이어야 한다: {args.grid}")

    channel = ScreenQr(ScreenQrConfig.from_yaml(args.channel_config))
    fountain_config = LtConfig.from_yaml(args.fountain_config)
    if args.chunk_size is not None:
        # 파일 전송처럼 큰 페이로드는 조각을 크게 잡아야 한다(5.2절: 클수록 유리).
        # yaml을 고치지 않고 실험 조건으로 바꿀 수 있게 인자로 둔다. 송수신이 같아야 한다.
        fountain_config = replace(fountain_config, chunk_size_bytes=args.chunk_size)
    payload, payload_label = _load_payload(args)
    chunks = split_into_chunks(payload, fountain_config.chunk_size_bytes)
    k = len(chunks)

    per_screen = qr_grid.droplets_per_screen(args.grid)
    log.info(
<<<<<<< HEAD
        "설정: payload=%s k=%d chunk_size_bytes=%d max_frames=%d display_ms=%s "
        "grid=%dx%d (화면당 방울 %d개)",
        payload_label,
=======
        "설정: message=%r k=%d chunk_size_bytes=%d max_frames=%d display_ms=%s "
        "grid=%dx%d (화면당 방울 %d개)",
        args.message,
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
        k,
        fountain_config.chunk_size_bytes,
        args.max_frames,
        channel.config.display_ms,
        args.grid,
        args.grid,
        per_screen,
    )
    if args.out_dir is None:
        # 화면 표시 모드에서만 시점을 맞춘다. 파일로 써내는 경로에는
        # 맞출 상대가 없으므로 기다리지 않는다.
        input(
            "Enter를 누르면 QR 프레임을 순서대로 띄우기 시작합니다 (수신측을 먼저 실행해둘 것)..."
        )

    def build_screens(start_seed: int) -> tuple[list, int]:
        """화면 max_frames장을 만든다. 화면 하나에 방울 per_screen개가 들어간다."""
        seed = start_seed
        screens = []
        for _ in range(args.max_frames):
            images = []
            for _ in range(per_screen):
                droplet = encode_droplet(chunks, seed, fountain_config)
                frame_bytes = frame.build_frame(seed=droplet.seed, payload=droplet.payload)
                images.append(channel.modulate(bytes_to_bits(frame_bytes)))
                seed += 1
            screens.append(qr_grid.tile(images, args.grid) if args.grid > 1 else images[0])
        return screens, seed

    if args.out_dir is not None:
        # 안드로이드 중계 경로: 화면을 띄우지 않고 시행별 동영상으로 써낸다.
        digits = max(2, len(str(args.trials)))
        # 시행마다 시드를 이어서 쓴다 — 같은 방울만 반복해 보내면 시행 간
        # 차이가 사라져 반복 측정의 의미가 없어진다.
        next_seed = 0
        for trial in range(1, args.trials + 1):
            screens, next_seed = build_screens(next_seed)
            path = args.out_dir / f"trial_{trial:0{digits}d}.mp4"
            written = screen_video.write_video(screens, path, channel.config.display_ms)
            log.info(
                "[%d/%d] 저장: %s (화면 %d장, %d프레임)",
                trial,
                args.trials,
                path,
                len(screens),
                written,
            )
        log.info(
            "완료: 동영상 %d개를 %s에 저장. 폰 A에서 전체화면·최대 밝기·자동회전 끄고 순서대로"
            " 재생하고, 폰 B로 녹화한 뒤 그 파일들을 screen_qr_recv.py --in-dir로 넘길 것"
<<<<<<< HEAD
            " (수신측에도 같은 --message 또는 --file, 그리고 --grid %d 를 넘길 것)",
            args.trials,
            args.out_dir,
=======
            " (--message %r, --grid %d 를 똑같이 맞출 것)",
            args.trials,
            args.out_dir,
            args.message,
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
            args.grid,
        )
        return

    next_seed = 0
    try:
        screens, next_seed = build_screens(0)
        for screen_index, screen in enumerate(screens):
            channel.emit(screen)
            if (screen_index + 1) % 10 == 0:
                log.info("%d/%d 화면 표시", screen_index + 1, args.max_frames)
    finally:
        # emit()은 창을 열어만 두고 안 닫는다(연속 표시 중 깜빡임 방지) — 여기서 한 번만 닫는다.
        channel.close_display()

    log.info(
        "완료: %d개 화면에 방울 %d개를 표시했다 (k=%d, grid=%dx%d)",
        args.max_frames,
        next_seed,
        k,
        args.grid,
        args.grid,
    )


if __name__ == "__main__":
    main()
