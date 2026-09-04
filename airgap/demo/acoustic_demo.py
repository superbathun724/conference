"""음향 FSK 단방향 발표용 데모 (M5 Tier 1, `docs/DEMO_UI_PLAN.md`의 SOUNDCHAT류 패널).

**왜 채팅이 아니라 단방향인가.** `acoustic_chat.py`는 두 기기가 서로 타이핑하는
양방향 데모라 양쪽 모두에서 파이썬이 돌아야 한다. 노트북이 한 대뿐이면 그
구성을 만들 수 없다. 이 데모는 대신 **폰이 미리 구운 WAV를 재생하고 노트북이
받는** 구성을 쓴다 — 폰은 계산을 전혀 하지 않고 스피커 노릇만 하므로 앱도
페이지도 필요 없고, 폰을 에어플레인 모드로 두면 "네트워크를 안 썼다"가 눈으로
증명된다. `guide.txt` 방법 B(WAV 파일 경유)와 같은 발상이다.

**측정 코드가 아니다.** `experiments/acoustic_send.py` / `acoustic_recv.py`가
시행을 반복하고 `data/raw/`에 CSV를 남기는 측정용이라면, 이 파일은 발표장에서
한 번 돌려 보여주기 위한 것이다. 그래서 아무것도 저장하지 않는다(CLAUDE.md 규칙 3).
판정 로직은 `channels/acoustic_fsk.py`의 `demodulate`와 `demo/acoustic_chat.py`의
`ChatReceiver`를 그대로 재사용한다 — 데모용으로 따로 손본 복조는 없다(규칙 5).

**스레드가 없다.** `acoustic_chat.py`는 타이핑을 받으면서 동시에 들어야 해서
백그라운드 스레드가 필요했지만, 이 데모는 듣기만 하므로 한 줄짜리 루프면 된다.
설명할 것이 하나 줄어드는 것이 발표 관점에서 이득이다.

사용법:
    # 1) 노트북에서 폰이 재생할 WAV를 굽는다
    python -m airgap.demo.acoustic_demo --mode bake --message "안녕하세요" \
        --message "에어갭 시연입니다" --out out/demo.wav

    # 2) out/demo.wav를 폰으로 옮겨 (에어플레인 모드로) 재생하고, 노트북에서
    python -m airgap.demo.acoustic_demo --mode recv --source mic

    # 마이크가 없거나 발표장에서 실패하면 — 미리 녹음해둔 파일로 같은 화면을 재생
    python -m airgap.demo.acoustic_demo --mode recv --source backup.wav
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from scipy.signal import resample_poly

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.core import frame
from airgap.core.bits import bytes_to_bits
from airgap.demo.acoustic_chat import ChatReceiver
from airgap.demo.demo_hub import DemoHub
from airgap.demo.hud_rich import render_level_bar, render_panel

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "channels" / "acoustic_fsk.yaml"
)


def build_demo_signal(
    channel: AcousticFsk, messages: list[str], gap_s: float = 1.0
) -> tuple[np.ndarray, list[int]]:
    """여러 메시지를 사이에 침묵을 두고 하나의 신호로 이어붙인다.

    파일 하나에 메시지를 여러 개 담아두면 발표 중에 폰을 다시 만질 일이 없다.
    메시지 사이의 침묵은 수신측이 앞 메시지의 끝과 다음 메시지의 프리앰블을
    섞어 읽지 않게 해주는 여유 구간이다 — 너무 짧으면 두 프레임이 한 버퍼 안에
    들어와 앞의 것만 읽히고 뒤의 것이 묻힌다.

    반환: (이어붙인 신호, 메시지별로 쓴 시드 목록)
    """
    gap = np.zeros(round(gap_s * channel.config.sample_rate_hz), dtype=np.float64)
    pieces: list[np.ndarray] = []
    seeds: list[int] = []
    for index, text in enumerate(messages, start=1):
        frame_bytes = frame.build_frame(seed=index, payload=text.encode("utf-8"))
        pieces.append(channel.modulate(bytes_to_bits(frame_bytes)))
        pieces.append(gap)
        seeds.append(index)
    return np.concatenate(pieces) if pieces else np.zeros(0), seeds


def _chunks_from_file(
    path: Path, sample_rate_hz: int, chunk_s: float, realtime: bool = True
) -> Iterator[np.ndarray]:
    """녹음 파일을 채널 표본율에 맞춘 뒤 조각으로 잘라 하나씩 내놓는다.

    폰 녹음 앱은 흔히 스테레오·48kHz로 저장한다. 복조는 심볼 길이를 표본율
    기준으로 계산하므로 표본율이 다르면 판정 자체가 어긋난다.

    `realtime`이면 조각 하나를 내놓을 때마다 그 길이만큼 기다린다. 파일을 최대
    속도로 쏟아내면 10초짜리 녹음이 0.1초 만에 끝나서, 백업 경로로 시연할 때
    화면이 실시간 수신처럼 보이지 않는다. 판정 결과는 기다리든 안 기다리든
    똑같다 — 보여주는 속도만 달라진다.

    (`experiments/acoustic_recv.py`의 `_load_signal`과 같은 처리를 한다. 데모가
    측정 스크립트를 import하지 않도록 일부러 따로 뒀다 — 나중에 세 번째 사용처가
    생기면 `core/`로 옮기는 게 맞다.)
    """
    signal, file_rate_hz = sf.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if file_rate_hz != sample_rate_hz:
        signal = resample_poly(signal, sample_rate_hz, file_rate_hz)

    step = max(1, round(chunk_s * sample_rate_hz))
    for start in range(0, len(signal), step):
        chunk = signal[start : start + step]
        yield chunk
        if realtime:
            time.sleep(len(chunk) / sample_rate_hz)


def _chunks_from_mic(sample_rate_hz: int, chunk_s: float) -> Iterator[np.ndarray]:
    """마이크를 열고 chunk_s초 분량씩 계속 읽어 내놓는다.

    `sounddevice`는 이 데모를 파일로만 돌릴 때는 필요 없으므로 여기서 import한다 —
    노트북 오디오 드라이버가 고장난 상태에서도 `--source 파일.wav` 경로는 그대로
    동작하게 하기 위해서다.
    """
    import sounddevice as sd

    block = max(1, round(chunk_s * sample_rate_hz))
    with sd.InputStream(samplerate=sample_rate_hz, channels=1, blocksize=block) as stream:
        while True:
            data, _overflowed = stream.read(block)
            yield data[:, 0].copy()


def rms(chunk: np.ndarray) -> float:
    """조각의 실효값. 레벨 막대에만 쓰는 값이라 판정에는 전혀 관여하지 않는다."""
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk))))


def channel_info_line(channel: AcousticFsk) -> str:
    """패널 맨 위에 띄울 채널 설정 한 줄. 발표 중 질문이 가장 많이 나오는 값들이다."""
    config = channel.config
    return (
        f"채널   FSK {config.freq0_hz / 1000:.1f}/{config.freq1_hz / 1000:.1f} kHz"
        f" · 심볼 {config.symbol_duration_ms:.0f} ms"
        f" · {channel.caps.nominal_bps:.0f} bps"
    )


def run_bake(args: argparse.Namespace, console: Console) -> None:
    """폰이 재생할 WAV를 만든다. 이 단계는 발표 전에 미리 해둔다."""
    channel = AcousticFsk(AcousticFskConfig.from_yaml(args.config))
    signal, seeds = build_demo_signal(channel, args.message, gap_s=args.gap_s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, signal, channel.config.sample_rate_hz)

    duration_s = len(signal) / channel.config.sample_rate_hz
    hub = DemoHub(channel_name="acoustic_fsk")
    hub.set_mode("TX")
    hub.update(info_lines=(channel_info_line(channel),))
    for text, seed in zip(args.message, seeds, strict=True):
        hub.log(f"seed={seed} · {text}")
    hub.update(banner=f"{args.out} 저장 — {len(args.message)}개 메시지, {duration_s:.1f}s")
    console.print(render_panel(hub.snapshot()))


def run_recv(args: argparse.Namespace, console: Console) -> None:
    """마이크나 파일에서 소리를 받아 메시지를 뽑고, 그 과정을 패널로 보여준다."""
    channel = AcousticFsk(AcousticFskConfig.from_yaml(args.config))
    receiver = ChatReceiver(channel, max_buffer_s=args.buffer_s)

    hub = DemoHub(channel_name="acoustic_fsk")
    hub.set_mode("RX")
    hub.update(info_lines=(channel_info_line(channel), "수신   대기 중"))
    console.print(render_panel(hub.snapshot()))

    sample_rate_hz = channel.config.sample_rate_hz
    if args.source == "mic":
        chunks = _chunks_from_mic(sample_rate_hz, args.chunk_s)
    else:
        chunks = _chunks_from_file(
            Path(args.source), sample_rate_hz, args.chunk_s, realtime=args.realtime
        )

    received = 0
    attempts = 0
    misses = 0
    # 시계가 아니라 "지금까지 들은 소리의 길이"로 간격을 잰다. 마이크에서는 둘이
    # 거의 같지만, 파일을 최대 속도로 읽을 때는 시계 기준이면 파일 전체가 순식간에
    # 지나가면서 복조를 한 번밖에 못 해본다.
    audio_elapsed_s = 0.0
    last_decode_at = 0.0
    last_panel_at = 0.0

    for chunk in chunks:
        audio_elapsed_s += len(chunk) / sample_rate_hz
        if audio_elapsed_s > args.timeout_s:
            hub.log(f"{args.timeout_s:.0f}초가 지나 종료한다")
            break

        receiver.append(chunk)

        # 프리앰블 상관은 버퍼 전체에 대한 계산이라 조각마다 돌리면 CPU가 못 버틴다
        # (acoustic_chat.py의 백그라운드 리스너와 같은 이유) — 간격을 둔다.
        if audio_elapsed_s - last_decode_at >= args.decode_interval_s:
            last_decode_at = audio_elapsed_s
            attempts += 1
            text = receiver.try_decode()
            if text is None:
                misses += 1
            else:
                received += 1
                hub.log(f"수신: {text}")
                # 방금 읽은 소리를 버퍼에서 비운다 — 안 그러면 다음 메시지가 와도
                # 계속 앞 프레임만 다시 찾는다 (ChatReceiver.reset_buffer 설명 참고)
                receiver.reset_buffer()

        hub.update(
            level_bar=render_level_bar(rms(chunk)),
            info_lines=(
                channel_info_line(channel),
                f"수신   메시지 {received}개 · 디코딩 시도 {attempts} · 못 찾음 {misses}",
            ),
        )
        if audio_elapsed_s - last_panel_at >= args.panel_interval_s:
            console.print(render_panel(hub.snapshot()))
            last_panel_at = audio_elapsed_s

    hub.set_mode("IDLE")
    hub.update(
        level_bar="", banner=f"수신 종료 — 메시지 {received}개 / 소리 {audio_elapsed_s:.1f}s"
    )
    console.print(render_panel(hub.snapshot()))


def main() -> None:
    parser = argparse.ArgumentParser(description="음향 FSK 단방향 발표용 데모")
    parser.add_argument("--mode", choices=["bake", "recv"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--message",
        action="append",
        default=None,
        help="[bake] 보낼 메시지. 여러 번 쓰면 한 파일에 순서대로 담는다",
    )
    parser.add_argument("--out", type=Path, default=Path("out/demo.wav"), help="[bake] 저장 경로")
    parser.add_argument("--gap-s", type=float, default=1.0, help="[bake] 메시지 사이 침묵 길이")
    parser.add_argument("--source", default="mic", help="[recv] 'mic' 또는 녹음 파일 경로")
    parser.add_argument("--chunk-s", type=float, default=0.2, help="[recv] 한 번에 읽는 길이(초)")
    parser.add_argument("--buffer-s", type=float, default=8.0, help="[recv] 판정에 쓰는 최근 길이")
    parser.add_argument("--decode-interval-s", type=float, default=0.3, help="[recv] 복조 간격")
    parser.add_argument("--panel-interval-s", type=float, default=0.5, help="[recv] 패널 갱신 간격")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="[recv] 최대 대기 시간")
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="[recv] 파일을 실제 재생 속도로 기다리지 않고 최대 속도로 읽는다 (확인용)",
    )
    args = parser.parse_args()

    if args.message is None:
        args.message = ["AIRGAP 시연입니다"]

    console = Console()
    if args.mode == "bake":
        run_bake(args, console)
    else:
        run_recv(args, console)


if __name__ == "__main__":
    main()
