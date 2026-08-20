"""LT 파운틴 부호 · 조각 유실률별 복원 오버헤드 측정.

물리 채널이 전혀 필요 없는 순수 알고리즘 실험이다 (core/fountain.py만 사용).
방울(조각)을 유실률만큼 무작위로 버리면서 몇 개를 받아야 K개 원본을 복원할
수 있는지를 반복 측정한다. "오버헤드"는 복원에 실제로 쓰인 방울 수 ÷ K다 —
1.0이면 이론적 최솟값과 같고, 클수록 낭비가 많다는 뜻이다.

사용법:
    python -m airgap.experiments.fountain_overhead --repeats 20
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from airgap.core.fountain import LtConfig, LtDecoder, encode_droplet, split_into_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "fountain.yaml"
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_LOSS_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _run_trial(chunks: list[bytes], config: LtConfig, loss_rate: float, seed: int, max_k_mult: int):
    """방울을 유실률만큼 버리면서 하나씩 먹여, 복원 성공까지 걸린 방울 수를 잰다."""
    k = len(chunks)
    loss_rng = np.random.default_rng(seed)
    decoder = LtDecoder(k, config)

    droplet_seed = 0
    delivered = 0
    max_generated = max_k_mult * k
    while not decoder.is_complete and droplet_seed < max_generated:
        if loss_rng.random() >= loss_rate:  # 유실률보다 크면 통과, 작으면 버림
            droplet = encode_droplet(chunks, droplet_seed, config)
            decoder.add_droplet(droplet.seed, droplet.payload)
            delivered += 1
        droplet_seed += 1

    return decoder.is_complete, delivered, droplet_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="LT 파운틴 부호 유실률별 복원 오버헤드 측정")
    parser.add_argument("--payload-bytes", type=int, default=640, help="원본 데이터 크기(바이트)")
    parser.add_argument("--repeats", type=int, default=20, help="유실률 조건당 반복 횟수")
    parser.add_argument(
        "--loss-rates", type=float, nargs="+", default=DEFAULT_LOSS_RATES, help="시험할 유실률 목록"
    )
    parser.add_argument(
        "--max-k-mult", type=int, default=8, help="복원 실패 판정 전 최대로 만들어볼 방울 수(K배수)"
    )
    parser.add_argument("--seed", type=int, default=0, help="원본 데이터 생성용 시드")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="LT 부호 설정 YAML 경로"
    )
    args = parser.parse_args()

    config = LtConfig.from_yaml(args.config)
    master_rng = np.random.default_rng(args.seed)
    data = master_rng.integers(0, 256, size=args.payload_bytes, dtype=np.uint8).tobytes()
    chunks = split_into_chunks(data, config.chunk_size_bytes)
    k = len(chunks)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_RAW_DIR / f"{run_id}_fountain_overhead"
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "trials.csv"

    log.info(
        "설정: run_id=%s k=%d chunk_size_bytes=%d c=%s delta=%s loss_rates=%s repeats=%d seed=%d",
        run_id,
        k,
        config.chunk_size_bytes,
        config.c,
        config.delta,
        args.loss_rates,
        args.repeats,
        args.seed,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["loss_rate", "trial", "k", "decoded", "droplets_delivered", "droplets_generated"]
        )

        for loss_rate in args.loss_rates:
            successes = 0
            for trial in range(1, args.repeats + 1):
                trial_seed = int(master_rng.integers(0, 2**32 - 1))
                decoded, delivered, generated = _run_trial(
                    chunks, config, loss_rate, trial_seed, args.max_k_mult
                )
                if decoded:
                    successes += 1
                writer.writerow([loss_rate, trial, k, int(decoded), delivered, generated])

            log.info(
                "유실률 %.0f%%: %d/%d 복원 성공",
                loss_rate * 100,
                successes,
                args.repeats,
            )

    log.info("완료: %s", csv_path)


if __name__ == "__main__":
    main()
