"""화면 밝기 변조 채널 실측 시연 · 수신측.

screen_flicker_send.py와 짝을 이룬다. 카메라로 화면을 촬영하며 평균
밝기를 fps 주기로 표본화하고(channels/screen_flicker.py의 capture()), 그
밝기 신호 하나를 통째로 복조해서 프레임 하나를 얻는다 — 파운틴 방울을
여러 개 모으는 QR 채널과 달리, 음향 채널처럼 시행마다 프레임을 한 번에
주고받는 방식이다.

사용법 1 — 이 스크립트가 직접 카메라를 여는 기기(웹캠 달린 노트북 등):
    python -m airgap.experiments.screen_flicker_recv --trials 10 --distance-cm 30

사용법 2 — 수신측을 안드로이드 폰으로 대신할 때 (M1의 WAV 파일 경유,
M3 QR의 동영상 경유와 같은 발상 — 폰 카메라 앱으로 화면을 녹화해 노트북으로
옮긴 뒤 그 동영상 파일을 읽는다):
    python -m airgap.experiments.screen_flicker_recv --in-dir in/recv \
        --distance-cm 30 --expected-payload "AIRGAP 20자 테스트문자열"

   --in-dir 안의 파일은 이름순(trial_01.*, trial_02.* ...)으로 읽는다.
   폰이 찍은 동영상의 실제 프레임 속도가 config의 fps와 다를 수 있으므로,
   동영상 메타데이터의 fps를 읽어 그 값으로 복조한다(음향 채널이 표본율
   차이를 리샘플링으로 맞추는 것과 같은 이유 — 무엇을 바꿨는지 로그로
   남긴다).
"""

import argparse
import csv
import logging
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from airgap.channels.screen_flicker import ScreenFlicker, ScreenFlickerConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "channels" / "screen_flicker.yaml"
DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def _trim_to_bytes(bits: np.ndarray) -> np.ndarray:
    return bits[: len(bits) - len(bits) % 8]


def _evaluate(channel: ScreenFlicker, signal: np.ndarray, expected_payload: bytes | None):
    """밝기 신호 하나를 복조해서 (성공 여부, 복원 문자열, 실패 사유)를 돌려준다.

    acoustic_recv.py의 _evaluate와 판정 기준을 맞춰야 두 채널의 성공률을
    공정하게 비교할 수 있다(CLAUDE.md 규칙 5 — 채널 간 비교는 공정해야 한다).
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


def _sample_video_file(path: Path, config: ScreenFlickerConfig) -> tuple[np.ndarray, float]:
    """녹화된 동영상 파일을 읽어 프레임별 평균 밝기 배열과 실제 fps를 돌려준다.

    live capture()가 fps 목표 주기로 프레임을 골라내는 것과 달리, 이미 저장된
    파일은 그 안의 프레임이 곧 표본이라고 본다 — 동영상 자체의 fps 메타데이터가
    실제 촬영 fps에 가장 가깝기 때문이다.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"동영상 파일을 열 수 없다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = config.fps
        log.warning("%s: fps 메타데이터를 못 읽어 설정값(%s)을 그대로 씀", path.name, config.fps)
    samples = []
    try:
        while True:
            ok, cam_frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
            samples.append(float(gray.mean()) / 255.0)
    finally:
        cap.release()
    return np.array(samples, dtype=np.float64), fps


def main() -> None:
    parser = argparse.ArgumentParser(description="화면 밝기 변조 수신 시연/실측")
    parser.add_argument("--trials", type=int, default=10, help="반복 횟수 (--in-dir 사용 시 무시)")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=40.0,
        help="시행당 촬영 길이(초) — 심볼 하나가 100ms(기본값)라 음향 채널보다 훨씬 느리다",
    )
    parser.add_argument(
        "--distance-cm", type=float, required=True, help="화면-카메라 거리(cm), 기록용"
    )
    parser.add_argument(
        "--expected-payload",
        default=None,
        help="송신측 --message와 비교할 원문 (생략 시 CRC 통과만 확인)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="채널 설정 YAML 경로"
    )
    parser.add_argument("--camera-index", type=int, default=None, help="카메라 장치 인덱스")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="지정하면 카메라 대신 이 폴더의 동영상 파일을 이름순으로 읽어 판정한다 (안드로이드용)",
    )
    args = parser.parse_args()

    config = ScreenFlickerConfig.from_yaml(args.config)
    if args.camera_index is not None:
        config = replace(config, camera_index=args.camera_index)
    channel = ScreenFlicker(config)
    expected_payload = args.expected_payload.encode("utf-8") if args.expected_payload else None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = (
        "screen_flicker_manual_relay" if args.in_dir is not None else "screen_flicker_manual"
    )
    run_dir = DATA_RAW_DIR / f"{run_id}_{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "trials.csv"

    if args.in_dir is not None:
        trial_files = sorted(p for p in args.in_dir.iterdir() if p.is_file())
        if not trial_files:
            raise SystemExit(f"{args.in_dir}에 파일이 없다")
        trials = len(trial_files)
        log.info(
            "설정(동영상 경유): run_id=%s in_dir=%s(절대경로: %s) trials=%d distance_cm=%s",
            run_id,
            args.in_dir,
            args.in_dir.resolve(),
            trials,
            args.distance_cm,
        )
        log.info("config=%s", args.config)
        # 폴더를 잘못 지정해 예전 파일을 다시 읽는 실수를 실행 도중 바로 알아챌 수 있게,
        # 처리 전에 각 파일의 실제 수정 시각을 로그로 나열한다
        # (2026-08-21 음향 채널 --distance-cm 오기재 사고 참고, acoustic_recv.py와 동일).
        for p in trial_files:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            log.info("  - %s (수정 시각 %s, %d bytes)", p.name, mtime, p.stat().st_size)
        # 측정 원본 보존(CLAUDE.md 규칙 3) — 실제로 쓴 동영상을 run_dir 밑에 복사해 둔다.
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
                samples, video_fps = _sample_video_file(source_path, config)
                if abs(video_fps - config.fps) > 0.5:
                    log.warning(
                        "%s: 동영상 fps %.2f != 채널 설정 %.2f → 이 시행만 fps %.2f로 복조",
                        source_path.name,
                        video_fps,
                        config.fps,
                        video_fps,
                    )
                trial_channel = ScreenFlicker(replace(config, fps=video_fps))
                signal = samples
                source_name = source_path.name
            else:
                input(f"[{trial}/{trials}] Enter로 촬영 시작 (송신측과 타이밍 맞추기)...")
                trial_channel = channel
                signal = channel.capture(args.duration_s)
                source_name = ""

            success, decoded_text, reason = _evaluate(trial_channel, signal, expected_payload)
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
