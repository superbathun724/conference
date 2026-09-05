"""화면 QR 채널 실측 시연 · 수신측.

screen_qr_send.py와 짝을 이룬다. 카메라로 화면을 계속 촬영하면서 매
프레임 QR 인식을 시도하고, 인식에 성공한 방울을 파운틴 디코더에 먹인다.
K개 조각이 모두 확정되면(core/fountain.py의 LtDecoder.is_complete) 그
자리에서 바로 끝낸다 — 이후 프레임은 더 볼 필요가 없다.

LT 부호는 "원본을 몇 개 조각으로 나눴는지(K)"를 미리 알아야 디코딩을
시작할 수 있다. 실전 프로토콜이라면 K를 헤더에 실어 보내겠지만, 여기서는
--message로 송신측과 똑같은 문자열을 넘겨 같은 방식(같은 chunk_size)으로
K를 계산하는 것으로 단순화했다 — 송신측 --message와 반드시 같아야 한다.

송신측이 --grid로 QR을 여러 장 동시에 띄워도 이 스크립트는 바꿀 것이 없다.
화면을 격자대로 잘라 나누지 않고, 한 프레임에서 zbar가 찾아낸 QR을 전부
받아(ScreenQr.demodulate_all) 방울로 넣기 때문이다 — 자르는 방식은 화면이
카메라 화각의 어디에 걸렸는지에 의존하지만 이 방식은 그런 가정이 없다.
--grid는 CSV에 조건을 남기기 위한 기록용 인자이며, 송신측 값과 맞춰 적어야
한다(--distance-cm과 같은 성격이다).

사용법 1 — 이 스크립트가 직접 카메라를 여는 기기(웹캠 달린 노트북 등):
    python -m airgap.experiments.screen_qr_recv --trials 5 --distance-cm 30 \
        --message "AIRGAP 20자 테스트문자열"

사용법 2 — 수신측을 안드로이드 폰으로 대신할 때 (이 프로젝트의 파이썬
코드는 폰에서 돌지 않으므로, 폰 카메라 앱으로 화면을 녹화한 뒤 그 동영상
파일을 노트북으로 옮겨 읽는다 — M1의 WAV 파일 경유(acoustic_recv.py
--in-dir)와 같은 발상):
    1) 폰으로 시행마다 짧은 동영상을 찍는다(화면이 QR을 다 보여줄 때까지).
       trial_01.mp4, trial_02.mp4 ... 순서로 이름을 맞춰 한 폴더에 모은다.
    2) python -m airgap.experiments.screen_qr_recv --in-dir in/recv \
           --distance-cm 30 --message "AIRGAP 20자 테스트문자열"
   --in-dir 안의 파일은 이름순으로 읽고, 각 파일이 시행 하나에 대응한다.
   cv2.VideoCapture는 카메라 장치 번호든 동영상 파일 경로든 똑같이
   받아들이므로, 프레임을 하나씩 꺼내 판정하는 코드는 실시간 카메라
   경로와 동일하다 — 이 스크립트 안에서 소스만 갈아끼운 것뿐이다.
"""

import argparse
import csv
import logging
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes
from airgap.core.fountain import LtConfig, LtDecoder, split_into_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNEL_CONFIG_PATH = _PROJECT_ROOT / "config" / "channels" / "screen_qr.yaml"
DEFAULT_FOUNTAIN_CONFIG_PATH = _PROJECT_ROOT / "config" / "fountain.yaml"
DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DEFAULT_MESSAGE = "AIRGAP 20자 테스트문자열"  # noqa: RUF001 (연구 목적의 한글 테스트 문자열)


def _frames_from_camera(camera_index: int | None, duration_s: float):
    """실시간 카메라에서 duration_s 동안 프레임을 그레이스케일로 하나씩 내놓는다."""
    index = camera_index if camera_index is not None else 0
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없다 (camera_index={index})")
    try:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            ok, cam_frame = cap.read()
            if ok:
                yield cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def _frames_from_video_file(path: Path):
    """녹화된 동영상 파일(폰 카메라로 화면을 찍은 것)을 처음부터 끝까지 읽는다.

    cv2.VideoCapture는 카메라 장치 번호 대신 파일 경로를 줘도 똑같이
    동작한다 — 실시간이 아니라 이미 저장된 프레임을 순서대로 돌려줄
    뿐이다. 그래서 프레임을 판정하는 쪽 코드는 실시간 카메라 경로와
    하나도 다르지 않다.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"동영상 파일을 열 수 없다: {path}")
    try:
        while True:
            ok, cam_frame = cap.read()
            if not ok:
                return  # 파일 끝
            yield cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def _receive_one_trial(
    channel: ScreenQr, k: int, fountain_config: LtConfig, frames
) -> tuple[LtDecoder, int, int, int]:
    """프레임을 하나씩 받아 방울을 모으고, 복원되면 바로 멈춘다.

    frames는 실시간 카메라든 동영상 파일이든 그레이스케일 프레임을
    내놓는 이터러블이면 된다 — 소스가 무엇이든 판정 기준은 같아야
    두 경로(실기기 카메라 vs 폰 동영상 경유)의 성공률을 공정하게
    비교할 수 있다.

    한 프레임에 QR이 여러 장 찍혀 있을 수 있으므로(송신측 --grid) 찾아낸
    심볼을 전부 방울로 넣는다. grid=1이면 심볼이 한 장뿐이라 예전과 동작이
    같다.

    반환: (디코더, 받은 서로 다른 방울 수, 처리한 프레임 수, 읽은 QR 심볼 수)
    """
    decoder = LtDecoder(k, fountain_config)
    frames_seen = 0
    symbols_seen = 0
    seen_seeds: set[int] = set()
    for gray in frames:
        if decoder.is_complete:
            break
        frames_seen += 1
        for bits in channel.demodulate_all(gray):
            symbols_seen += 1
            parsed = frame.parse_frame(bits_to_bytes(bits))
            if parsed is None:
                continue
            seen_seeds.add(parsed.seed)
            decoder.add_droplet(parsed.seed, parsed.payload)

    return decoder, len(seen_seeds), frames_seen, symbols_seen


def main() -> None:
    parser = argparse.ArgumentParser(description="화면 QR 수신 시연/실측")
    parser.add_argument("--trials", type=int, default=5, help="반복 횟수")
    parser.add_argument("--duration-s", type=float, default=60.0, help="시행당 최대 촬영 시간(초)")
    parser.add_argument(
        "--distance-cm", type=float, required=True, help="화면-카메라 거리(cm), 기록용"
    )
    parser.add_argument(
        "--message", default=DEFAULT_MESSAGE, help="송신측 --message와 똑같이 맞출 원문"
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
    parser.add_argument("--camera-index", type=int, default=None, help="카메라 장치 인덱스")
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
        help="송신측이 --file로 보낸 원본 파일. K 계산과 복원 대조에 쓴다",
    )
    parser.add_argument(
=======
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
        "--grid",
        type=int,
        default=1,
        help="송신측 --grid와 같은 값(기록용). 판정 동작에는 영향이 없다",
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help=(
            "지정하면 실시간 카메라 대신 이 폴더의 동영상 파일을 이름순으로 읽어 "
            "시행마다 하나씩 처리한다 (안드로이드 폰으로 화면을 녹화해 옮긴 파일용)"
        ),
    )
    args = parser.parse_args()

    channel = ScreenQr(ScreenQrConfig.from_yaml(args.channel_config))
    fountain_config = LtConfig.from_yaml(args.fountain_config)
    if args.chunk_size is not None:
        # 파일 전송처럼 큰 페이로드는 조각을 크게 잡아야 한다(5.2절: 클수록 유리).
        # yaml을 고치지 않고 실험 조건으로 바꿀 수 있게 인자로 둔다. 송수신이 같아야 한다.
        fountain_config = replace(fountain_config, chunk_size_bytes=args.chunk_size)
    if args.file is not None:
        payload = args.file.read_bytes()
        payload_label = f"{args.file.name} ({len(payload)} B)"
    else:
        payload = args.message.encode("utf-8")
        payload_label = f"message={args.message!r}"
    chunks = split_into_chunks(payload, fountain_config.chunk_size_bytes)
    k = len(chunks)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = "screen_qr_manual_relay" if args.in_dir is not None else "screen_qr_manual"
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
<<<<<<< HEAD
        log.info("k=%d payload=%s grid=%dx%d", k, payload_label, args.grid, args.grid)
=======
        log.info("k=%d message=%r grid=%dx%d", k, args.message, args.grid, args.grid)
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
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
<<<<<<< HEAD
            "설정: run_id=%s trials=%d duration_s=%s distance_cm=%s k=%d payload=%s grid=%dx%d",
=======
            "설정: run_id=%s trials=%d duration_s=%s distance_cm=%s k=%d message=%r grid=%dx%d",
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
            run_id,
            trials,
            args.duration_s,
            args.distance_cm,
            k,
<<<<<<< HEAD
            payload_label,
=======
            args.message,
>>>>>>> 8d68fdcdc1abbdf089a6ee4b2bbe3993cab1d85d
            args.grid,
            args.grid,
        )

    successes = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial",
                "distance_cm",
                "k",
                "success",
                "decoded_payload",
                "droplets_received",
                "frames_seen",
                "qr_symbols_seen",
                "grid",
                "reason",
                "source_file",
            ]
        )

        for trial in range(1, trials + 1):
            if trial_files is not None:
                source_path = trial_files[trial - 1]
                frames = _frames_from_video_file(source_path)
                source_name = source_path.name
            else:
                input(f"[{trial}/{trials}] Enter로 촬영 시작 (송신측 화면을 카메라로 비출 것)...")
                frames = _frames_from_camera(args.camera_index, args.duration_s)
                source_name = ""

            decoder, droplets_received, frames_seen, symbols_seen = _receive_one_trial(
                channel, k, fountain_config, frames
            )

            if decoder.is_complete:
                restored = decoder.assemble(len(payload))
                success = restored == payload
                reason = "" if success else "복원됐으나 원문과 다름"
                try:
                    decoded_text = restored.decode("utf-8")
                except UnicodeDecodeError:
                    # 이미지 같은 이진 파일은 내용을 찍어 봐야 읽을 수 없다. 크기만 남긴다.
                    match = "일치" if success else "불일치"
                    decoded_text = f"<이진 {len(restored)} B, 원본과 {match}>"
            else:
                success = False
                decoded_text = ""
                reason = "제한 시간 내 방울 부족"

            if success:
                successes += 1
            log.info(
                "[%d/%d] %s (프레임 %d개에서 QR %d장 읽어 방울 %d개 수신)",
                trial,
                trials,
                "성공: " + decoded_text if success else "실패: " + reason,
                frames_seen,
                symbols_seen,
                droplets_received,
            )
            writer.writerow(
                [
                    trial,
                    args.distance_cm,
                    k,
                    int(success),
                    decoded_text,
                    droplets_received,
                    frames_seen,
                    symbols_seen,
                    args.grid,
                    reason,
                    source_name,
                ]
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
