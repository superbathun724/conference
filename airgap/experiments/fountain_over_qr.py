"""LT 파운틴 부호를 화면 QR 채널에 실제로 얹어보는 통합 실험.

`fountain_overhead.py`는 파운틴 부호만 떼어놓고 조각 유실을 숫자로 흉내 낸
순수 알고리즘 실험이었다. 이 스크립트는 그 방울을 실제로 core/frame.py
프레임에 담고 channels/screen_qr.py로 진짜 QR 이미지를 만들었다가 다시
읽어서, 파운틴층·프레임층·물리층 전체를 한 번에 통과시켜본다.

카메라는 쓰지 않는다 — channel.modulate()로 QR 이미지를 만들고 곧바로
channel.demodulate()로 다시 읽는 것도 엄연한 루프백이다(docs/ARCHITECTURE.md
"루프백 모드" 참고). "화면에 매 프레임마다 다른 QR 코드가 뜨고, 그중 일부를
카메라가 놓친다"는 상황을 유실률로 흉내 낸다: 방울을 건너뛰면 카메라가 그
프레임을 놓친 것이고, 건너뛰지 않은 방울은 실제로 QR 이미지를 만들고
zbar로 다시 읽어 정말로 복원되는지까지 확인한다(순수 XOR 계산만 하는
fountain_overhead.py보다 이 스크립트가 실제 채널에 한 걸음 더 가깝다).

사용법:
    python -m airgap.experiments.fountain_over_qr --repeats 20
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.core.fountain import LtConfig, LtDecoder, encode_droplet, split_into_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOUNTAIN_CONFIG_PATH = _PROJECT_ROOT / "config" / "fountain.yaml"
DEFAULT_CHANNEL_CONFIG_PATH = _PROJECT_ROOT / "config" / "channels" / "screen_qr.yaml"
DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DEFAULT_LOSS_RATES = [0.0, 0.1, 0.2, 0.3]


def _run_trial(
    chunks: list[bytes],
    original_length: int,
    fountain_config: LtConfig,
    channel: ScreenQr,
    loss_rate: float,
    seed: int,
    max_frames: int,
) -> tuple[bool, int, int, int]:
    """방울을 하나씩 QR로 만들었다 다시 읽으면서, 일부는 유실률만큼 건너뛴다.

    반환: (복원 성공 여부, 실제 전달된 방울 수, 시도한 프레임 수, QR 인식 자체가 실패한 횟수)
    """
    k = len(chunks)
    loss_rng = np.random.default_rng(seed)
    decoder = LtDecoder(k, fountain_config)

    delivered = 0
    frames_tried = 0
    qr_decode_failures = 0
    droplet_seed = 0

    while not decoder.is_complete and frames_tried < max_frames:
        frames_tried += 1
        if loss_rng.random() < loss_rate:
            droplet_seed += 1
            continue  # 카메라가 이 프레임을 놓쳤다고 가정 — QR을 만들지도 않는다

        droplet = encode_droplet(chunks, droplet_seed, fountain_config)
        frame_bytes = frame.build_frame(seed=droplet.seed, payload=droplet.payload)
        bits = bytes_to_bits(frame_bytes)

        image = channel.modulate(bits)
        recovered_bits = channel.demodulate(image)

        if len(recovered_bits) < len(bits):
            qr_decode_failures += 1  # 유실이 아니라 QR 생성·인식 자체가 깨진 경우
            droplet_seed += 1
            continue

        parsed = frame.parse_frame(bits_to_bytes(recovered_bits[: len(bits)]))
        if parsed is not None:
            decoder.add_droplet(parsed.seed, parsed.payload)
            delivered += 1
        droplet_seed += 1

    if decoder.is_complete:
        restored = decoder.assemble(original_length)
        original = b"".join(chunks)[:original_length]
        if restored != original:
            raise AssertionError("복원된 바이트열이 원본과 다르다 — 파운틴 디코더 버그")

    return decoder.is_complete, delivered, frames_tried, qr_decode_failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="파운틴 부호 + 화면 QR 채널 통합 실험 (카메라 불필요)"
    )
    parser.add_argument("--payload-bytes", type=int, default=160, help="원본 데이터 크기(바이트)")
    parser.add_argument("--repeats", type=int, default=20, help="유실률 조건당 반복 횟수")
    parser.add_argument(
        "--loss-rates", type=float, nargs="+", default=DEFAULT_LOSS_RATES, help="시험할 유실률 목록"
    )
    parser.add_argument(
        "--max-frames", type=int, default=200, help="복원 실패로 판정하기 전 최대 시도 프레임 수"
    )
    parser.add_argument("--seed", type=int, default=0, help="원본 데이터 생성용 시드")
    parser.add_argument(
        "--fountain-config",
        type=Path,
        default=DEFAULT_FOUNTAIN_CONFIG_PATH,
        help="LT 부호 설정 YAML",
    )
    parser.add_argument(
        "--channel-config", type=Path, default=DEFAULT_CHANNEL_CONFIG_PATH, help="QR 채널 설정 YAML"
    )
    args = parser.parse_args()

    fountain_config = LtConfig.from_yaml(args.fountain_config)
    channel = ScreenQr(ScreenQrConfig.from_yaml(args.channel_config))

    master_rng = np.random.default_rng(args.seed)
    data = master_rng.integers(0, 256, size=args.payload_bytes, dtype=np.uint8).tobytes()
    chunks = split_into_chunks(data, fountain_config.chunk_size_bytes)
    k = len(chunks)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_RAW_DIR / f"{run_id}_fountain_over_qr"
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "trials.csv"

    log.info(
        "설정: run_id=%s k=%d chunk_size_bytes=%d loss_rates=%s repeats=%d seed=%d",
        run_id,
        k,
        fountain_config.chunk_size_bytes,
        args.loss_rates,
        args.repeats,
        args.seed,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "loss_rate",
                "trial",
                "k",
                "decoded",
                "droplets_delivered",
                "frames_tried",
                "qr_decode_failures",
            ]
        )

        for loss_rate in args.loss_rates:
            successes = 0
            for trial in range(1, args.repeats + 1):
                trial_seed = int(master_rng.integers(0, 2**32 - 1))
                decoded, delivered, frames_tried, qr_failures = _run_trial(
                    chunks,
                    len(data),
                    fountain_config,
                    channel,
                    loss_rate,
                    trial_seed,
                    args.max_frames,
                )
                if decoded:
                    successes += 1
                writer.writerow(
                    [loss_rate, trial, k, int(decoded), delivered, frames_tried, qr_failures]
                )

            log.info("유실률 %.0f%%: %d/%d 복원 성공", loss_rate * 100, successes, args.repeats)

    log.info("완료: %s", csv_path)


if __name__ == "__main__":
    main()
