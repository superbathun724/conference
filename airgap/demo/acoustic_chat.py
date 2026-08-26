"""실시간 음향 채팅 데모 (M5 발표 시연용).

SOUNDCHAT 같은 앱(폰 두 대가 소리로 실시간 채팅하는 것)에서 영감을 얻었다.
물리층은 그대로다 — `channels/acoustic_fsk.py`를 한 글자도 안 바꾸고 그
위에 "타이핑하면 바로 내보내고, 백그라운드로 항상 듣고 있다가 메시지가
오면 바로 찍는" 껍데기만 씌운 것뿐이다.

M1~M4의 실험 스크립트(acoustic_send/recv.py)는 시행 하나 = 메시지 하나를
사람이 Enter로 동기화해서 보내고 판정하는 방식이다. BER·처리율을 정확히
재는 데는 그 방식이 맞지만, 발표 시연에는 딱딱하다. 이 데모는 측정용이
아니라 "이 채널로 이런 것도 됩니다"를 보여주는 용도다.

전이중(동시에 말하고 듣기)이 아니라 반이중이다. 같은 기기에서 스피커가
소리를 내는 동안 그 기기의 마이크도 그 소리를 그대로 들어버리기 때문에
(자기 반향) — 내가 방금 보낸 (시드, 내용)을 기억해뒀다가, 그게 다시
들리면 "받은 메시지"가 아니라 자기 반향으로 걸러낸다.

사용법 (두 대의 기기에서 각각 실행, PowerShell에서 한글이 깨지면 먼저 `chcp 65001`):
    python -m airgap.demo.acoustic_chat
그 다음 메시지를 입력하고 Enter. 상대가 보낸 메시지는 백그라운드에서
자동으로 화면에 뜬다. Ctrl+C로 종료.

Tier 1 데모 UI(`docs/DEMO_UI_PLAN.md`): 상태는 `DemoHub`가 쥐고 있고, 메시지를
보내거나 받을 때마다 `hud_rich.render_panel()`로 그 상태를 다시 그려서 출력한다
(항상 갱신되는 화면이 아니라 "사건이 생길 때마다 새로 찍는" 방식 — 이유는
hud_rich.py 상단 설명 참고).
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits
from airgap.demo.demo_hub import DemoHub
from airgap.demo.hud_rich import render_panel

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "channels" / "acoustic_fsk.yaml"
)


def _trim_to_bytes(bits: np.ndarray) -> np.ndarray:
    return bits[: len(bits) - len(bits) % 8]


class ChatReceiver:
    """수신 오디오를 계속 누적하면서, 새로 들어온(자기 반향이 아닌) 메시지를 골라낸다.

    스레드·오디오 장치와는 분리해뒀다 — feed()에 numpy 배열만 넣어주면
    되므로, 실제 마이크 없이도 이 클래스 하나만 테스트할 수 있다
    (tests/test_acoustic_chat.py 참고).
    """

    def __init__(self, channel: AcousticFsk, max_buffer_s: float = 8.0) -> None:
        self._channel = channel
        self._max_buffer_len = round(max_buffer_s * channel.config.sample_rate_hz)
        self._buffer = np.zeros(0, dtype=np.float64)
        # (시드, 내용) 쌍으로 "이미 처리한 프레임"을 기억한다. 시드만으로 구분하면
        # 두 기기가 각자 1부터 시드를 매기다가 우연히 같은 시드를 써서 서로 다른
        # 메시지가 충돌하는 경우를 놓친다 — 그래서 반드시 내용까지 같이 본다.
        self._handled: set[tuple[int, bytes]] = set()

    def mark_own(self, seed: int, payload: bytes) -> None:
        """방금 내가 보낸 (시드, 내용)을 기록해, 자기 반향으로 다시 들려도 무시하게 한다."""
        self._handled.add((seed, payload))

    def append(self, chunk: np.ndarray) -> None:
        """새 오디오 조각을 버퍼에 쌓는다 (디코딩은 시도하지 않음, 오디오 콜백마다 부른다).

        오디오를 놓치지 않으려면 콜백이 올 때마다 바로 쌓아야 하지만, 디코딩(프리앰블
        상관 계산)까지 매번 하면 느리다 — 그래서 쌓는 것과 디코딩 시도를 분리했다.
        """
        self._buffer = np.concatenate([self._buffer, chunk])
        if len(self._buffer) > self._max_buffer_len:
            self._buffer = self._buffer[-self._max_buffer_len :]

    def feed(self, chunk: np.ndarray) -> str | None:
        """append() + try_decode()를 한 번에 — 테스트나 간단한 사용에 편의상 둔 것."""
        self.append(chunk)
        return self.try_decode()

    def try_decode(self) -> str | None:
        bits = self._channel.demodulate(self._buffer)
        if len(bits) == 0:
            return None
        parsed = frame.parse_frame(bits_to_bytes(_trim_to_bytes(bits)))
        if parsed is None:
            return None

        key = (parsed.seed, parsed.payload)
        if key in self._handled:
            return None
        self._handled.add(key)

        try:
            return parsed.payload.decode("utf-8")
        except UnicodeDecodeError:
            return repr(parsed.payload)


def _run_background_listener(
    receiver: ChatReceiver,
    sample_rate_hz: int,
    stop_event: threading.Event,
    decode_interval_s: float,
    hub: DemoHub,
    console: Console,
) -> None:
    """마이크를 계속 열어두고, 일정 간격으로만 디코딩을 시도하는 백그라운드 루프.

    매 오디오 콜백(수십 ms 단위)마다 프리앰블 상관(_find_preamble_start)을
    다시 계산하면 몇 초 분량 버퍼에 대한 상관 계산 비용이 계속 쌓여 CPU가
    못 버틴다. 그래서 오디오 자체는 콜백에서 큐에 계속 쌓아두되(놓치는
    구간이 없게), 디코딩 시도만 decode_interval_s 간격으로 제한한다.
    """
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, frames_count, time_info, status):  # noqa: ARG001
        audio_queue.put(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=sample_rate_hz, channels=1, callback=_callback)
    stream.start()
    try:
        last_decode_at = 0.0
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            receiver.append(chunk)

            now = time.monotonic()
            if now - last_decode_at < decode_interval_s:
                continue
            last_decode_at = now

            text = receiver.try_decode()
            if text is not None:
                hub.log(f"수신: {text}")
                console.print(render_panel(hub.snapshot()))
                print("> ", end="", flush=True)
    finally:
        stream.stop()
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="실시간 음향 채팅 데모 (M5 시연용)")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="채널 설정 YAML 경로"
    )
    parser.add_argument(
        "--buffer-s", type=float, default=8.0, help="수신 판정에 쓰는 최근 오디오 길이(초)"
    )
    parser.add_argument(
        "--decode-interval-s",
        type=float,
        default=0.3,
        help="프리앰블 상관 재계산 최소 간격(초) — CPU 부하 제한용",
    )
    args = parser.parse_args()

    config = AcousticFskConfig.from_yaml(args.config)
    channel = AcousticFsk(config)
    receiver = ChatReceiver(channel, max_buffer_s=args.buffer_s)

    console = Console()
    hub = DemoHub(channel_name="acoustic_fsk")
    hub.set_mode("RX")  # 항상 듣고 있으므로 기본 상태는 RX

    stop_event = threading.Event()
    listener = threading.Thread(
        target=_run_background_listener,
        args=(receiver, config.sample_rate_hz, stop_event, args.decode_interval_s, hub, console),
        daemon=True,
    )
    listener.start()

    console.print(render_panel(hub.snapshot()))
    print("메시지를 입력하고 Enter, Ctrl+C로 종료")
    seed = 1
    try:
        while True:
            text = input("> ")
            if not text:
                continue
            payload = text.encode("utf-8")
            frame_bytes = frame.build_frame(seed=seed, payload=payload)
            bits = bytes_to_bits(frame_bytes)
            signal = channel.modulate(bits)

            receiver.mark_own(seed, payload)  # 재생 전에 먼저 등록 — 자기 반향 대비
            hub.set_mode("TX")
            hub.log(f"보냄: {text}")
            console.print(render_panel(hub.snapshot()))
            channel.emit(signal)
            hub.set_mode("RX")  # 재생이 끝나면 다시 듣는 상태로 복귀

            seed = (seed + 1) % 0x10000
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\n종료")


if __name__ == "__main__":
    main()
