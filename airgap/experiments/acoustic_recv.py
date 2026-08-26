"""음향 채널 실측 시연 · 수신측.

acoustic_send.py와 짝을 이룬다. 시행마다 녹음 → 복조 → 프레임 해체까지 하고,
전체 시행이 끝나면 성공률을 출력한다. 결과는 data/raw/에 시행별로 남긴다
(원본 데이터는 이후 수정하지 않는다 — CLAUDE.md 규칙).

사용법 1 — 노트북 마이크로 직접 녹음:
    python -m airgap.experiments.acoustic_recv --trials 10 --duration-s 8 --distance-cm 100

사용법 2 — 안드로이드 폰으로 녹음한 파일을 대신 넘길 때 (노트북 오디오
드라이버가 고장났거나 장치가 없을 때의 대체 경로. acoustic_send.py의
--out-dir로 만든 WAV를 폰 A에서 재생하고, 폰 B로 1m 거리에서 녹음한 뒤
그 결과 파일들을 한 폴더에 모아 --in-dir로 넘긴다):
    python -m airgap.experiments.acoustic_recv --in-dir in/recv --distance-cm 100 \
        --expected-payload "AIRGAP 20자 테스트문자열"

--in-dir 안의 파일은 이름순(trial_01.*, trial_02.* ...)으로 읽는다. 폰
녹음 앱이 WAV로 저장하지 않으면(m4a/aac 등은 sounddevice가 아니라
soundfile이 읽는데, soundfile은 특허 코덱을 지원하지 않는다) 미리
ffmpeg로 변환해야 한다: `ffmpeg -i in.m4a -ar 44100 -ac 1 out.wav`
"""

import argparse
import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "channels" / "acoustic_fsk.yaml"
)
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _trim_to_bytes(bits):
    return bits[: len(bits) - len(bits) % 8]


def _evaluate(channel: AcousticFsk, signal: np.ndarray, expected_payload: bytes | None):
    """신호 하나를 복조해서 (성공 여부, 복원 문자열, 실패 사유)를 돌려준다.

    라이브 녹음 경로와 파일 경유 경로가 판정 기준을 공유해야 두 방식의
    성공률을 공정하게 비교할 수 있어서 하나로 뽑아냈다.
    """
    bits = channel.demodulate(signal)
    if len(bits) == 0:
        return False, "", "프리앰블 미검출"

    parsed = frame.parse_frame(bits_to_bytes(_trim_to_bytes(bits)))
    if parsed is None:
        return False, "", "CRC 불일치"
    if expected_payload is not None and parsed.payload != expected_payload:
        return False, "", "페이로드 불일치"

    try:
        decoded_text = parsed.payload.decode("utf-8")
    except UnicodeDecodeError:
        decoded_text = repr(parsed.payload)
    return True, decoded_text, ""


def _load_signal(path: Path, target_sample_rate_hz: int) -> np.ndarray:
    """녹음 파일을 읽어 채널의 표본율에 맞춘 모노 신호로 돌려준다.

    폰 녹음 앱은 흔히 스테레오·48kHz로 저장한다. 복조는 심볼 길이·FFT
    구간을 표본율에 맞춰 계산하므로, 표본율이 다르면 리샘플링으로 맞추지
    않는 한 판정 자체가 어긋난다. 무엇을 바꿨는지 반드시 로그로 남긴다
    (CLAUDE.md: 측정 데이터를 손대지 않되, 손댄 사실은 숨기지 않는다).
    """
    signal, sample_rate_hz = sf.read(path)
    if signal.ndim > 1:
        log.warning("%s: 스테레오 %d채널 → 평균으로 모노 변환", path.name, signal.shape[1])
        signal = signal.mean(axis=1)
    if sample_rate_hz != target_sample_rate_hz:
        log.warning(
            "%s: 녹음 표본율 %dHz != 채널 설정 %dHz → 리샘플링",
            path.name,
            sample_rate_hz,
            target_sample_rate_hz,
        )
        signal = resample_poly(signal, target_sample_rate_hz, sample_rate_hz)
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="음향 FSK 수신 시연/실측")
    parser.add_argument("--trials", type=int, default=10, help="반복 횟수 (--in-dir 사용 시 무시)")
    parser.add_argument("--duration-s", type=float, default=8.0, help="시행당 녹음 길이(초)")
    parser.add_argument(
        "--distance-cm", type=float, required=True, help="스피커-마이크 거리(cm), 기록용"
    )
    parser.add_argument(
        "--expected-payload",
        default=None,
        help="송신측 --message와 비교할 원문 (생략 시 CRC 통과만 확인)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="채널 설정 YAML 경로"
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="지정하면 마이크 녹음 대신 이 폴더의 파일을 이름순으로 읽어 판정한다 (안드로이드용)",
    )
    args = parser.parse_args()

    config = AcousticFskConfig.from_yaml(args.config)
    channel = AcousticFsk(config)
    expected_payload = args.expected_payload.encode("utf-8") if args.expected_payload else None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = "acoustic_manual_relay" if args.in_dir is not None else "acoustic_manual"
    run_dir = DATA_RAW_DIR / f"{run_id}_{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "trials.csv"

    if args.in_dir is not None:
        trial_files = sorted(p for p in args.in_dir.iterdir() if p.is_file())
        if not trial_files:
            raise SystemExit(f"{args.in_dir}에 파일이 없다")
        trials = len(trial_files)
        log.info(
            "설정(파일 경유): run_id=%s in_dir=%s(절대경로: %s) trials=%d distance_cm=%s config=%s",
            run_id,
            args.in_dir,
            args.in_dir.resolve(),
            trials,
            args.distance_cm,
            args.config,
        )
        # 폴더를 잘못 찾아 예전 파일을 다시 읽는 실수를 실행 도중에 바로 알아챌 수 있게,
        # 처리 전에 각 파일의 실제 수정 시각을 눈에 보이게 나열한다.
        for p in trial_files:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            log.info("  - %s (수정 시각 %s, %d bytes)", p.name, mtime, p.stat().st_size)
        # 원본 파일은 data/raw 바깥(in_dir)에 있어 언제든 사용자가 지우거나 덮어쓸 수 있다.
        # 이 실행에 실제로 어떤 원본이 쓰였는지 나중에 검증할 수 있도록 run_dir 밑에 복사해 둔다
        # (CLAUDE.md 규칙: 측정 원본은 보존한다).
        archive_dir = run_dir / "src"
        archive_dir.mkdir()
        for p in trial_files:
            shutil.copy2(p, archive_dir / p.name)
    else:
        trial_files = None
        trials = args.trials
        log.info(
            "설정: run_id=%s trials=%d duration_s=%s distance_cm=%s config=%s",
            run_id,
            trials,
            args.duration_s,
            args.distance_cm,
            args.config,
        )

    successes = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["trial", "distance_cm", "success", "decoded_payload", "reason", "source_file"]
        )

        for trial in range(1, trials + 1):
            if trial_files is not None:
                source_path = trial_files[trial - 1]
                signal = _load_signal(source_path, config.sample_rate_hz)
                source_name = source_path.name
            else:
                input(f"[{trial}/{trials}] Enter로 녹음 시작 (송신측과 타이밍 맞추기)...")
                signal = channel.capture(args.duration_s)
                source_name = ""

            success, decoded_text, reason = _evaluate(channel, signal, expected_payload)
            if success:
                successes += 1
            log.info(
                "[%d/%d] %s",
                trial,
                trials,
                "성공: " + decoded_text if success else "실패: " + reason,
            )
            writer.writerow(
                [trial, args.distance_cm, int(success), decoded_text, reason, source_name]
            )

    success_rate = successes / trials
    log.info(
        "완료: %d/%d 성공 (성공률 %.0f%%) -> %s",
        successes,
        trials,
        success_rate * 100,
        csv_path,
    )


if __name__ == "__main__":
    main()
