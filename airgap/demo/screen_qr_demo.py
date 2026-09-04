"""화면 QR 전송 발표용 데모 (M5 Tier 1, `docs/DEMO_UI_PLAN.md`의 DECIMEN류 통계 패널).

**실험 코드가 아니다.** `experiments/screen_qr_send.py` / `screen_qr_recv.py`가
측정용(시행 반복, `data/raw/`에 CSV 기록)이라면, 이 파일은 발표장에서 한 번
돌려 보여주기 위한 것이다. 그래서 측정 데이터를 저장하지 않는다(CLAUDE.md 규칙 3).
복원한 파일만 `--out-dir`에 쓴다.

**채널 코드는 건드리지 않는다.** `channels/screen_qr.py`의 `modulate` /
`demodulate` / `emit`을 그대로 호출하기만 한다. 화면에 뿌리는 숫자(초당 프레임
수, 유실 수, 실효 전송률)는 전부 이 파일에서 세는 것이라, 특정 채널만 데모를
위해 최적화되는 일이 없다(CLAUDE.md 규칙 5).

**무엇이든 보낼 수 있다 — 다만 크기 제한이 실질적이다.** 채널은 바이트열만
다루므로 이미지든 오디오든 상관없다. 실제 속도는 이렇게 정해진다:

    한 프레임에 담기는 양 = 조각 크기 (최대 255바이트, core/frame.py의 길이 칸이 8비트)
    초당 프레임 수        = 1000 / display_ms
    실효 전송률           ≈ 조각 크기 × fps ÷ 파운틴 오버헤드(1.5~2.0배)

기본값(조각 192B, display_ms 200)이면 대략 초당 0.5~0.6KB다. 10KB 이미지는
약 30초, 50KB는 2분 반. **동영상·오디오 원본(수 MB)은 몇 시간이 걸려 현실적이지
않다** — 발표에서 다루려면 몇 초짜리 저해상도 클립이나 썸네일로 줄여야 한다.
DECIMEN 영상의 140KB/s는 폰 화면 60fps와 폰 카메라 60fps가 맞물린 값이라,
웹캠 30fps인 이 구성과는 조건 자체가 다르다.

사용법 — 노트북 화면(송신) + 카메라(수신):
    (수신측 먼저) python -m airgap.demo.screen_qr_demo --mode recv --out-dir out\recv
    (송신측)      python -m airgap.demo.screen_qr_demo --mode send --file cat.png

    # 문자열만 보낼 때
    python -m airgap.demo.screen_qr_demo --mode send --message "AIRGAP 시연"

카메라가 없거나 발표장에서 실시간 인식이 실패할 때 — 미리 찍어둔 동영상으로
같은 화면을 재생할 수 있다 (PLAN.md M5 "시연 설계 원칙: 실시간 측정에 의존하지
않는다"):
    python -m airgap.demo.screen_qr_demo --mode recv --source backup.mp4

`--source`는 숫자면 카메라 장치 번호, 아니면 동영상 파일 경로다. `cv2.VideoCapture`가
둘을 똑같이 받아들이므로 프레임을 꺼내 판정하는 코드는 두 경우가 완전히 같다.

**수신측은 무엇이 오는지 미리 알 필요가 없다.** 송신측이 안내표 프레임을 주기적으로
섞어 보내고(`demo/manifest.py`), 수신측은 그것을 잡아 파일 이름·크기·조각 설정을
알아낸다.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from airgap.channels.screen_qr import ScreenQr, ScreenQrConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.core.fountain import LtConfig, LtDecoder, encode_droplet, split_into_chunks
from airgap.demo.demo_hub import DemoHub
from airgap.demo.hud_rich import render_panel
from airgap.demo.manifest import MANIFEST_SEED, Manifest, build_manifest, parse_manifest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNEL_CONFIG_PATH = _PROJECT_ROOT / "config" / "channels" / "screen_qr.yaml"
DEFAULT_FOUNTAIN_CONFIG_PATH = _PROJECT_ROOT / "config" / "fountain.yaml"
DEFAULT_MESSAGE = "AIRGAP 20자 테스트문자열"  # noqa: RUF001 (연구 목적의 한글 테스트 문자열)
MESSAGE_NAME = "message.txt"  # --message로 보낼 때 붙이는 이름
FILE_CHUNK_SIZE_BYTES = 192  # 파일 전송 기본 조각 크기 (프레임 길이 칸 상한 255보다 여유 있게)


class RateMeter:
    """최근 window_s초 안에 일어난 사건 수를 세어 "초당 몇 번"으로 바꾼다.

    시작부터의 누적 평균(전체 횟수 ÷ 전체 시간)을 쓰면 시간이 갈수록 숫자가
    굳어져서, 카메라가 화면을 놓치기 시작해도 화면상으로는 티가 안 난다.
    최근 몇 초만 보면 지금 상태가 그대로 드러난다.
    """

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._times: deque[float] = deque()

    def tick(self, now: float) -> None:
        self._times.append(now)

    def rate(self, now: float) -> float:
        cutoff = now - self.window_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        return len(self._times) / self.window_s


class QrReceiveSession:
    """카메라 프레임을 받아 안내표와 방울을 모으면서, 그 과정을 숫자로 세는 얇은 래퍼.

    복원 자체는 `LtDecoder`가 하고 QR 해독은 `ScreenQr`가 한다 — 이 클래스가
    직접 하는 일은 "무엇이 몇 번 일어났는지" 세는 것과, 안내표가 오기 전에 받은
    방울을 잠시 들고 있다가 나중에 몰아 넣는 것뿐이다. 카메라가 필요 없어서
    (프레임 이미지를 인자로 받으므로) 합성 이미지만으로 테스트할 수 있다.

    세는 항목의 뜻:
      - dropped   : QR을 못 찾았거나 프레임 검사(CRC)에 걸린 카메라 프레임.
                    초점이 안 맞거나 화면이 바뀌는 순간에 찍힌 것들이다.
      - dup       : 이미 본 시드. 화면 한 장이 여러 프레임에 걸쳐 찍히므로
                    자연스럽게 가장 큰 값이 된다 — 낭비가 아니라 정상이다.
      - redundant : 처음 보는 방울인데 확정 조각 수를 당장 늘리지 못한 것.
                    LT 부호는 차수 2 이상인 방울을 모아뒀다가 나중에 한꺼번에
                    푸는 구조라, 이 값이 크다고 손해는 아니다. 파운틴 부호의
                    오버헤드(K의 1.4~2.0배, FINDINGS.md 2026-08-20)가 화면에서
                    눈으로 보이는 지점이 바로 이 숫자다.
    """

    def __init__(self, channel: ScreenQr, manifest: Manifest | None = None) -> None:
        self.channel = channel
        self.manifest: Manifest | None = None
        self.decoder: LtDecoder | None = None
        self.frames_seen = 0
        self.frames_dropped = 0
        self.droplets_new = 0
        self.droplets_dup = 0
        self.droplets_redundant = 0
        self._seen_seeds: set[int] = set()
        # 안내표를 잡기 전에 들어온 방울들. 버리면 카메라를 늦게 들이댄 만큼
        # 고스란히 손해라, 안내표가 오는 순간 한꺼번에 디코더에 넣는다.
        self._early_droplets: list[tuple[int, bytes]] = []
        if manifest is not None:
            self._start(manifest)

    @property
    def is_complete(self) -> bool:
        return self.decoder is not None and self.decoder.is_complete

    @property
    def chunks_done(self) -> int:
        return self.decoder.decoded_count if self.decoder is not None else 0

    def _start(self, manifest: Manifest) -> None:
        """안내표를 받아 디코더를 만들고, 그동안 모아둔 방울을 밀어 넣는다."""
        self.manifest = manifest
        config = LtConfig(
            chunk_size_bytes=manifest.chunk_size_bytes, c=manifest.c, delta=manifest.delta
        )
        self.decoder = LtDecoder(manifest.k, config)
        for seed, payload in self._early_droplets:
            self.decoder.add_droplet(seed, payload)
        self._early_droplets.clear()

    def feed(self, gray: np.ndarray) -> str | None:
        """카메라 프레임 한 장을 처리한다. 로그로 남길 만한 일이 있으면 그 문구를 돌려준다."""
        self.frames_seen += 1

        bits = self.channel.demodulate(gray)
        if len(bits) == 0:
            self.frames_dropped += 1
            return None

        parsed = frame.parse_frame(bits_to_bytes(bits))
        if parsed is None:
            self.frames_dropped += 1
            return None

        if parsed.seed == MANIFEST_SEED:
            return self._handle_manifest(parsed.payload)
        return self._handle_droplet(parsed.seed, parsed.payload)

    def _handle_manifest(self, payload: bytes) -> str | None:
        if self.manifest is not None:
            self.droplets_dup += 1  # 안내표는 주기적으로 계속 오므로 두 번째부터는 중복
            return None
        manifest = parse_manifest(payload)
        if manifest is None:
            self.frames_dropped += 1
            return None
        self._start(manifest)
        return f"안내표 수신: {manifest.name} · {manifest.size_bytes}B · 조각 {manifest.k}개"

    def _handle_droplet(self, seed: int, payload: bytes) -> str | None:
        if seed in self._seen_seeds:
            self.droplets_dup += 1
            return None
        self._seen_seeds.add(seed)
        self.droplets_new += 1

        if self.decoder is None:
            self._early_droplets.append((seed, payload))
            return None  # 아직 안내표가 없어 K를 모른다 — 일단 들고만 있는다

        before = self.decoder.decoded_count
        self.decoder.add_droplet(seed, payload)
        gained = self.decoder.decoded_count - before
        if gained == 0:
            self.droplets_redundant += 1
            return None
        return f"방울 seed={seed} → 조각 +{gained}"

    def recovered_bytes(self) -> bytes:
        """복원된 원본. 아직 다 못 모았으면 예외가 난다."""
        if self.decoder is None or self.manifest is None:
            raise RuntimeError("안내표를 아직 받지 못했다")
        return self.decoder.assemble(self.manifest.size_bytes)


def _gray_frames(source: str, realtime: bool = True) -> Iterator[np.ndarray]:
    """카메라 또는 동영상 파일에서 흑백 프레임을 하나씩 내놓는다.

    source가 숫자 문자열이면 카메라 장치 번호로, 아니면 파일 경로로 본다.
    `cv2.VideoCapture`가 두 경우를 같은 방식으로 다루므로 이 아래는 완전히 같다.

    동영상 파일을 `realtime`으로 읽으면 그 영상의 초당 프레임 수에 맞춰 기다린다.
    최대 속도로 쏟아내면 30초짜리 백업 영상이 몇 초 만에 끝나서, 발표장에서
    실시간 수신처럼 보이지 않는다. 판정 결과는 기다리든 안 기다리든 똑같다.
    카메라는 원래 실시간으로 들어오므로 이 대기가 필요 없다.
    """
    is_camera = source.isdigit()
    handle: int | str = int(source) if is_camera else source
    cap = cv2.VideoCapture(handle)
    if not cap.isOpened():
        raise RuntimeError(
            f"영상 소스를 열 수 없다: {source!r}\n"
            "  - 카메라를 쓸 생각이었다면 --source 0 / 1 처럼 장치 번호를 바꿔 본다\n"
            "  - 발표장에서 카메라가 안 되면 --source 미리찍어둔영상.mp4 로 대체한다"
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    pace_s = 1.0 / video_fps if (realtime and not is_camera and video_fps > 0) else 0.0
    try:
        while True:
            started = time.monotonic()
            ok, bgr = cap.read()
            if not ok:
                return  # 파일 끝이거나 카메라가 끊겼다
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if pace_s:
                remaining = pace_s - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()


def load_payload(args: argparse.Namespace) -> tuple[bytes, str, int]:
    """무엇을 보낼지 정한다. 반환: (바이트열, 이름, 조각 크기)"""
    if args.file is not None:
        payload = args.file.read_bytes()
        chunk_size = args.chunk_size or FILE_CHUNK_SIZE_BYTES
        return payload, args.file.name, chunk_size
    fountain_config = LtConfig.from_yaml(args.fountain_config)
    chunk_size = args.chunk_size or fountain_config.chunk_size_bytes
    return args.message.encode("utf-8"), MESSAGE_NAME, chunk_size


def frame_plan(k: int, manifest_every: int, passes: float = 2.5) -> int:
    """몇 프레임을 띄울지 정한다.

    파운틴 부호는 K개를 복원하는 데 대략 K의 1.4~2.0배가 필요하다(FINDINGS.md
    2026-08-20). 되돌아갈 통신로가 없어 "다 받았다"는 신호를 받을 수 없으므로,
    여유를 둬서 2.5배로 잡고 그만큼 무조건 다 띄운다. 수신측이 그 전에 끝냈다면
    남은 프레임은 안 봐도 됐던 것뿐이다.
    """
    droplets = max(1, round(k * passes))
    manifests = droplets // manifest_every + 1
    return droplets + manifests


def run_send(args: argparse.Namespace, console: Console) -> None:
    """안내표와 파운틴 방울을 QR로 번갈아 띄우면서, 진행 상황을 패널로 보여준다."""
    channel_config = ScreenQrConfig.from_yaml(args.channel_config)
    if args.display_ms is not None:
        channel_config = replace_display_ms(channel_config, args.display_ms)
    channel = ScreenQr(channel_config)

    payload, name, chunk_size = load_payload(args)
    base_config = LtConfig.from_yaml(args.fountain_config)
    fountain_config = LtConfig(
        chunk_size_bytes=chunk_size, c=base_config.c, delta=base_config.delta
    )
    chunks = split_into_chunks(payload, chunk_size)
    manifest = Manifest(
        name=name,
        size_bytes=len(payload),
        chunk_size_bytes=chunk_size,
        c=fountain_config.c,
        delta=fountain_config.delta,
    )
    manifest_bytes = build_manifest(manifest)
    total_frames = args.max_frames or frame_plan(len(chunks), args.manifest_every)

    hub = DemoHub(channel_name="screen_qr")
    hub.set_mode("TX")
    hub.update(payload_bytes=len(payload), chunks_total=len(chunks))
    hub.log(f"보낼 것: {name} · {len(payload)}B → 조각 {len(chunks)}개 × {chunk_size}B")
    hub.log(f"프레임 {total_frames}개 예정 (안내표 {args.manifest_every}프레임마다)")
    console.print(render_panel(hub.snapshot()))

    input("Enter를 누르면 QR을 띄우기 시작합니다 (수신측을 먼저 켜둘 것)... ")

    display_rate = RateMeter()
    started = time.monotonic()
    droplet_seed = 1  # 시드 0은 안내표 전용
    shown = 0
    try:
        for index in range(total_frames):
            # 안내표를 주기적으로 섞는다 — 늦게 카메라를 들이댄 수신측도 곧 하나를 잡는다
            if index % args.manifest_every == 0:
                frame_bytes = frame.build_frame(seed=MANIFEST_SEED, payload=manifest_bytes)
            else:
                droplet = encode_droplet(chunks, droplet_seed, fountain_config)
                frame_bytes = frame.build_frame(seed=droplet.seed, payload=droplet.payload)
                droplet_seed += 1
            channel.emit(channel.modulate(bytes_to_bits(frame_bytes)))
            shown += 1

            now = time.monotonic()
            display_rate.tick(now)
            hub.update(frames_seen=shown, capture_fps=display_rate.rate(now))
            if shown % args.panel_every == 0:
                hub.log(f"{shown}/{total_frames} 프레임 표시")
                console.print(render_panel(hub.snapshot()))
    except KeyboardInterrupt:
        hub.log("사용자가 중단했다")
    finally:
        # emit()은 연속 표시 중 깜빡임을 막으려고 창을 열어만 둔다 — 여기서 한 번 닫는다.
        channel.close_display()

    elapsed = time.monotonic() - started
    hub.set_mode("IDLE")
    hub.update(banner=f"표시 완료 — {shown}프레임 in {elapsed:.1f}s")
    console.print(render_panel(hub.snapshot()))


def replace_display_ms(config: ScreenQrConfig, display_ms: float) -> ScreenQrConfig:
    """설정 파일은 그대로 두고 화면 갱신 간격만 바꾼 사본을 만든다.

    파일 전송은 문자열 하나보다 프레임이 훨씬 많이 필요해서 기본 500ms로는 너무
    오래 걸린다. 설정 YAML을 고치면 측정 스크립트까지 조건이 바뀌므로(CLAUDE.md
    규칙 5), 데모 실행에서만 덮어쓴다.
    """
    return ScreenQrConfig(
        box_size=config.box_size,
        border=config.border,
        error_correction=config.error_correction,
        display_ms=display_ms,
        fullscreen=config.fullscreen,
        camera_index=config.camera_index,
        capture_timeout_s=config.capture_timeout_s,
    )


def run_recv(args: argparse.Namespace, console: Console) -> None:
    """화면을 촬영해 안내표와 방울을 모으고, DECIMEN류 통계 패널을 계속 갱신한다."""
    channel = ScreenQr(ScreenQrConfig.from_yaml(args.channel_config))

    hub = DemoHub(channel_name="screen_qr")
    hub.set_mode("RX")
    hub.log(f"안내표를 기다리는 중 (source={args.source})")
    console.print(render_panel(hub.snapshot()))

    session = QrReceiveSession(channel)
    capture_rate = RateMeter()
    decode_rate = RateMeter()
    started = time.monotonic()
    last_panel_at = 0.0

    for gray in _gray_frames(args.source, realtime=args.realtime):
        now = time.monotonic()
        if now - started > args.timeout_s:
            hub.log(f"{args.timeout_s:.0f}초 안에 복원하지 못했다")
            break

        before_new = session.droplets_new
        note = session.feed(gray)
        capture_rate.tick(now)
        if session.droplets_new > before_new:
            decode_rate.tick(now)
        if note:
            hub.log(note)

        elapsed = now - started
        manifest = session.manifest
        hub.update(
            capture_fps=capture_rate.rate(now),
            decode_fps=decode_rate.rate(now),
            frames_seen=session.frames_seen,
            frames_dropped=session.frames_dropped,
            droplets_new=session.droplets_new,
            droplets_dup=session.droplets_dup,
            droplets_redundant=session.droplets_redundant,
            chunks_done=session.chunks_done,
            chunks_total=manifest.k if manifest else 0,
            payload_bytes=manifest.size_bytes if manifest else 0,
            goodput_kbps=goodput_kbps(
                session.chunks_done * (manifest.chunk_size_bytes if manifest else 0), elapsed
            ),
        )

        if session.is_complete:
            break
        if now - last_panel_at >= args.panel_interval_s:
            console.print(render_panel(hub.snapshot()))
            last_panel_at = now

    elapsed = time.monotonic() - started
    hub.set_mode("IDLE")
    if session.is_complete and session.manifest is not None:
        recovered = session.recovered_bytes()
        rate = goodput_kbps(len(recovered), elapsed)
        out_path = args.out_dir / session.manifest.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(recovered)
        hub.update(banner=f"전송 완료! — {len(recovered)} B in {elapsed:.1f}s ({rate:.2f} KB/s)")
        hub.log(f"저장: {out_path}")
        if session.manifest.name.endswith(".txt"):
            hub.log(f"내용: {recovered.decode('utf-8', errors='replace')}")
    elif session.manifest is None:
        hub.log("안내표를 한 번도 잡지 못했다 — 카메라 초점과 화면 밝기를 확인할 것")
    else:
        hub.log(f"복원 실패: {session.chunks_done}/{session.manifest.k} 조각까지만 확정")

    console.print(render_panel(hub.snapshot()))


def goodput_kbps(recovered_bytes: int, elapsed_s: float) -> float:
    """실효 전송률(KB/s). 아직 시간이 거의 안 흘렀으면 0으로 둔다(0으로 나누기 방지)."""
    if elapsed_s <= 0.0:
        return 0.0
    return recovered_bytes / 1024.0 / elapsed_s


def main() -> None:
    parser = argparse.ArgumentParser(description="화면 QR 전송 발표용 데모 (통계 패널 포함)")
    parser.add_argument("--mode", choices=["send", "recv"], required=True)
    parser.add_argument("--file", type=Path, default=None, help="[send] 보낼 파일 (이미지 등)")
    parser.add_argument(
        "--message", default=DEFAULT_MESSAGE, help="[send] --file이 없을 때 보낼 문자열"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="[send] 조각 크기(바이트). 최대 255"
    )
    parser.add_argument(
        "--display-ms", type=float, default=None, help="[send] QR 한 장을 띄워두는 시간"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="[send] 띄울 프레임 수 (기본: 조각 수 × 2.5)"
    )
    parser.add_argument(
        "--manifest-every", type=int, default=8, help="[send] 몇 프레임마다 안내표를 섞을지"
    )
    parser.add_argument(
        "--panel-every", type=int, default=10, help="[send] 몇 프레임마다 패널을 출력할지"
    )
    parser.add_argument(
        "--source", default="0", help="[recv] 카메라 장치 번호 또는 동영상 파일 경로"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out/recv"), help="[recv] 복원한 파일을 저장할 폴더"
    )
    parser.add_argument("--timeout-s", type=float, default=300.0, help="[recv] 최대 대기 시간")
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="[recv] 동영상 파일을 재생 속도로 기다리지 않고 최대 속도로 읽는다 (확인용)",
    )
    parser.add_argument(
        "--panel-interval-s", type=float, default=0.5, help="[recv] 패널을 다시 그리는 간격"
    )
    parser.add_argument("--channel-config", type=Path, default=DEFAULT_CHANNEL_CONFIG_PATH)
    parser.add_argument("--fountain-config", type=Path, default=DEFAULT_FOUNTAIN_CONFIG_PATH)
    args = parser.parse_args()

    if args.chunk_size is not None and not 1 <= args.chunk_size <= 255:
        raise SystemExit("조각 크기는 1~255바이트여야 한다 (core/frame.py의 길이 칸이 8비트)")

    console = Console()
    if args.mode == "send":
        run_send(args, console)
    else:
        run_recv(args, console)


if __name__ == "__main__":
    main()
