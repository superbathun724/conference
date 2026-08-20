"""음향 채널 실측 시연 · 송신측.

M1 완료 조건(스피커 1m 거리 마이크 무오류 복원, 10회 중 8회 이상)을 실제
장치로 확인하기 위한 스크립트. airgap/experiments/acoustic_recv.py와 짝을
이룬다.

사용법 1 — 노트북 스피커로 직접 재생 (두 노트북, 또는 한 노트북 + 외부
마이크로 1m 거리를 두고):
    (수신측 먼저) python -m airgap.experiments.acoustic_recv --trials 10
    (송신측)      python -m airgap.experiments.acoustic_send --trials 10

매 시행마다 Enter로 재생 시점을 맞춘다. 정밀한 자동 동기화 대신 사람이
"지금" 하고 맞추는 방식을 쓴 것은, 이 스크립트의 목적이 네트워크 동기화가
아니라 물리 채널 자체의 성능을 재는 것이기 때문이다.

사용법 2 — 노트북 오디오 장치가 없거나 고장났을 때, 안드로이드 두 대로 대체:
    python -m airgap.experiments.acoustic_send --trials 10 --out-dir out/send
이러면 스피커를 재생하는 대신 시행마다 WAV 파일(trial_01.wav ...)을
out-dir에 써낸다. 이 파일들을 폰 A로 옮겨 순서대로 재생하고, 폰 B로
1m 거리에서 녹음한 뒤 그 결과물을 acoustic_recv.py --in-dir로 넘긴다.
sounddevice(스피커/마이크)도, 노트북 오디오 드라이버도 건드리지 않으므로
드라이버가 고장난 상태에서도 실기기 검증을 진행할 수 있다.
"""

import argparse
import logging
from pathlib import Path

import soundfile as sf

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "channels" / "acoustic_fsk.yaml"
)
DEFAULT_MESSAGE = "AIRGAP 20자 테스트문자열"  # noqa: RUF001 (연구 목적의 한글 테스트 문자열)


def main() -> None:
    parser = argparse.ArgumentParser(description="음향 FSK 송신 시연/실측")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="보낼 문자열")
    parser.add_argument("--trials", type=int, default=10, help="반복 횟수")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="채널 설정 YAML 경로"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="지정하면 스피커 재생 대신 시행별 WAV 파일을 이 폴더에 써낸다 (안드로이드 중계용)",
    )
    args = parser.parse_args()

    config = AcousticFskConfig.from_yaml(args.config)
    channel = AcousticFsk(config)
    payload = args.message.encode("utf-8")

    log.info(
        "설정: message=%r trials=%d config=%s symbol_duration_ms=%s freq0=%s freq1=%s",
        args.message,
        args.trials,
        args.config,
        config.symbol_duration_ms,
        config.freq0_hz,
        config.freq1_hz,
    )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        digits = max(2, len(str(args.trials)))

    for trial in range(1, args.trials + 1):
        frame_bytes = frame.build_frame(seed=trial, payload=payload)
        bits = bytes_to_bits(frame_bytes)
        signal = channel.modulate(bits)

        if args.out_dir is not None:
            wav_path = args.out_dir / f"trial_{trial:0{digits}d}.wav"
            sf.write(wav_path, signal, config.sample_rate_hz)
            log.info("[%d/%d] 저장: %s", trial, args.trials, wav_path)
            continue

        input(
            f"[{trial}/{args.trials}] Enter를 누르면 재생합니다 (수신측 녹음 시작 후 누르세요)..."
        )
        channel.emit(signal)
        log.info("[%d/%d] 재생 완료", trial, args.trials)

    if args.out_dir is not None:
        log.info(
            "완료: %d개 WAV 파일을 %s에 저장. 폰 A로 옮겨 trial_%s 순서대로 재생하고,"
            " 폰 B(1m 거리)로 녹음한 뒤 그 파일들을 acoustic_recv.py --in-dir로 넘길 것"
            " (--expected-payload %r 도 함께 넘겨 원문과 비교)",
            args.trials,
            args.out_dir,
            "01".zfill(digits),
            args.message,
        )


if __name__ == "__main__":
    main()
