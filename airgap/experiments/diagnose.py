"""실기기 녹음(WAV)·녹화(MP4) 파일 하나를 넣으면 실패 원인을 숫자로 짚어 준다.

2026-09-05 실기기 측정에서 음향과 밝기 변조가 0/10이었다. 시뮬레이션으로 원인
후보를 좁혔지만(FINDINGS.md), 어느 것이 실제 원인인지는 촬영·녹음 원본을 봐야
확정된다. 이 도구는 그 확정을 위한 것이다. 판정 코드와 같은 함수를 쓰므로
"판정기가 무엇을 보았는가"를 그대로 보여 준다.

사용법:
    python -m airgap.experiments.diagnose in/acoustic/trial_01.wav
    python -m airgap.experiments.diagnose in/flicker/trial_01.mp4

각각 콘솔에 진단 표를 찍고, 같은 이름의 .png 그림을 옆에 저장한다.

읽는 법 (음향):
    - 최대 진폭이 0.05 아래면 볼륨·거리 문제다.
    - 프리앰블 점수(0~1)가 임계값 아래면 신호가 안 왔거나 동기 실패.
    - 추정 시계 오차가 ±0.2%를 넘으면 그것만으로 CRC가 깨진다 — 심볼 주기
      추정이 이를 흡수해야 하며, 그림의 '판정 또렷함'이 프레임 끝까지 유지되는지 본다.
    - 또렷함이 앞에서 뒤로 갈수록 떨어지면 시계 오차, 처음부터 낮으면 잔향·잡음.

읽는 법 (밝기 변조):
    - 두 준위 차이(위 포락선 − 아래 포락선)가 0.05 아래면 자동 노출이 진폭을
      눌렀거나 화면이 화각에서 너무 작다.
    - 프리앰블 NCC(−1~1)가 임계값 아래면 동기 실패.
    - 그림에서 밝기가 천천히 출렁이면(초 단위) 자동 노출 드리프트다.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from airgap.channels.acoustic_fsk import AcousticFsk, AcousticFskConfig
from airgap.channels.screen_flicker import ScreenFlicker, ScreenFlickerConfig
from airgap.core import frame
from airgap.core.bits import bits_to_bytes, bytes_to_bits

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _trim(bits: np.ndarray) -> np.ndarray:
    return bits[: (len(bits) // 8) * 8]


def _verdict(bits: np.ndarray) -> str:
    bits = _trim(bits)
    if len(bits) == 0:
        return "프리앰블 미검출"
    parsed = frame.parse_frame(bits_to_bytes(bits))
    if parsed is None:
        return "CRC 불일치"
    return f"성공 — {parsed.payload[:40]!r}"


# ────────────────────────── 음향 ──────────────────────────


def diagnose_wav(path: Path, out_png: Path) -> None:
    from airgap.experiments.acoustic_recv import _load_signal

    config = AcousticFskConfig.from_yaml(ROOT / "config/channels/acoustic_fsk.yaml")
    channel = AcousticFsk(config)
    signal = _load_signal(path, config.sample_rate_hz)
    fs = config.sample_rate_hz

    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal**2)))
    start = channel._find_preamble_start(signal)

    rows = [
        ("길이", f"{len(signal) / fs:.2f} 초"),
        ("최대 진폭", f"{peak:.3f}  (클리핑: {'예' if peak >= 0.99 else '아니오'})"),
        ("RMS", f"{rms:.4f}"),
        ("프리앰블 시작", "미검출" if start is None else f"{start / fs:.3f} 초 지점"),
    ]

    soft = None
    if start is not None:
        padded = np.concatenate([signal, np.zeros(channel._symbol_len)])
        period = channel._estimate_symbol_period(padded, start)
        nominal = float(channel._symbol_len)
        clock_pct = (period / nominal - 1.0) * 100.0
        soft = channel._soft_bits_at(padded, start, period)
        n_show = min(len(soft), 400)
        conf = np.abs(soft[:n_show])
        q1, q4 = conf[: n_show // 4].mean(), conf[3 * n_show // 4 : n_show].mean()
        clock_note = "CRC 깨질 수준" if abs(clock_pct) > 0.15 else "무시 가능"
        trend_note = "뒤로 갈수록 나빠짐 → 시계 오차 의심" if q4 < q1 - 0.15 else "유지됨"
        rows += [
            ("추정 심볼 주기", f"{period:.2f} 표본 (설정 {nominal:.0f})"),
            (
                "추정 시계 오차",
                f"{clock_pct:+.3f} %  ({clock_note})",
            ),
            ("판정 또렷함 (앞 1/4)", f"{q1:.2f}"),
            (
                "판정 또렷함 (뒤 1/4)",
                f"{q4:.2f}  ({trend_note})",
            ),
        ]
    rows.append(("최종 판정", _verdict(channel.demodulate(signal))))

    _print_table(path.name, rows)
    _plot_wav(signal, fs, start, soft, config, out_png)


def _plot_wav(signal, fs, start, soft, config, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_korean_font(plt)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    t = np.arange(len(signal)) / fs
    axes[0].plot(t, signal, lw=0.4)
    axes[0].set_title("파형 (진폭)")
    if start is not None:
        axes[0].axvline(start / fs, color="r", ls="--", label="프리앰블 시작")
        axes[0].legend()
    axes[0].set_xlabel("초")

    # 스펙트로그램 대용: 심볼 창 두 톤 에너지
    sym = round(config.symbol_duration_ms / 1000 * fs)
    hop = sym // 4
    n = (len(signal) - sym) // hop
    tt = np.arange(sym) / fs
    k0 = np.exp(-2j * np.pi * config.freq0_hz * tt)
    k1 = np.exp(-2j * np.pi * config.freq1_hz * tt)
    idx = (np.arange(n) * hop)[:, None] + np.arange(sym)[None, :]
    seg = signal[idx]
    e0, e1 = np.abs(seg @ k0), np.abs(seg @ k1)
    tx = (np.arange(n) * hop + sym / 2) / fs
    axes[1].plot(tx, e0, label=f"{config.freq0_hz:.0f} Hz (비트 0)", lw=0.8)
    axes[1].plot(tx, e1, label=f"{config.freq1_hz:.0f} Hz (비트 1)", lw=0.8)
    axes[1].set_title("두 반송 주파수의 에너지 (겹치면 잔향·간섭, 둘 다 낮으면 신호 없음)")
    axes[1].legend()
    axes[1].set_xlabel("초")

    if soft is not None:
        axes[2].plot(np.abs(soft), lw=0.8)
        axes[2].axhline(0.5, color="gray", ls=":")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("심볼별 판정 또렷함 |E1−E0|/(E1+E0)  — 뒤로 갈수록 떨어지면 시계 오차")
        axes[2].set_xlabel("심볼 번호")
    else:
        axes[2].text(
            0.5, 0.5, "프리앰블 미검출", ha="center", va="center", transform=axes[2].transAxes
        )
    fig.savefig(out_png, dpi=110)
    log.info("그림 저장: %s", out_png)


# ────────────────────────── 밝기 변조 ──────────────────────────


def diagnose_mp4(path: Path, out_png: Path) -> None:
    from airgap.experiments.screen_flicker_recv import _sample_video_file

    config = ScreenFlickerConfig.from_yaml(ROOT / "config/channels/screen_flicker.yaml")
    levels, fps = _sample_video_file(path, config)
    channel = ScreenFlicker(config)

    start = channel._find_preamble_start(levels)
    threshold = channel._local_threshold(levels)
    window = int(round(config.fps * config.drift_window_s)) or 1
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    upper = maximum_filter1d(levels, size=window, mode="nearest")
    lower = minimum_filter1d(levels, size=window, mode="nearest")
    swing = float(np.median(upper - lower))

    # NCC 점수 자체를 보여 준다
    pre = channel.modulate(bytes_to_bits(frame.PREAMBLE))
    tmpl = pre - pre.mean()
    n = len(tmpl)
    ones = np.ones(n)
    ws = np.convolve(levels, ones, "valid")
    wq = np.convolve(levels**2, ones, "valid")
    sd = np.sqrt(np.maximum(wq - ws**2 / n, 1e-12))
    ncc = np.correlate(levels, tmpl, "valid") / (sd * np.sqrt(np.sum(tmpl**2)))

    swing_note = "너무 작음 — 자동 노출 또는 화면이 화각에서 작음" if swing < 0.05 else "충분"
    rows = [
        ("프레임 수 / fps", f"{len(levels)} / {fps:.2f}"),
        ("전체 평균 밝기", f"{levels.mean():.3f}  (0=검정, 1=흰색)"),
        (
            "두 준위 차이 (포락선 폭, 중앙값)",
            f"{swing:.3f}  ({swing_note})",
        ),
        ("느린 드리프트 폭", f"{float(np.ptp(threshold)):.3f}  (기준값이 이만큼 움직였다)"),
        (
            "프리앰블 NCC 최댓값",
            f"{float(np.max(ncc)):.2f}  (임계값 {config.preamble_detect_threshold})",
        ),
        ("프리앰블 시작", "미검출" if start is None else f"프레임 {start} ({start / fps:.2f} 초)"),
        ("최종 판정", _verdict(channel.demodulate(levels))),
    ]
    _print_table(path.name, rows)
    _plot_mp4(levels, fps, upper, lower, threshold, start, out_png)


def _plot_mp4(levels, fps, upper, lower, threshold, start, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_korean_font(plt)

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    t = np.arange(len(levels)) / fps
    ax.plot(t, levels, lw=0.6, label="프레임 평균 밝기")
    ax.plot(t, upper, lw=0.8, ls="--", label="위 포락선")
    ax.plot(t, lower, lw=0.8, ls="--", label="아래 포락선")
    ax.plot(t, threshold, lw=1.0, color="k", label="판정 기준값(국소)")
    ax.axhline(0.5, color="gray", ls=":", label="예전 고정 기준 0.5")
    if start is not None:
        ax.axvline(start / fps, color="r", ls="--", label="프리앰블 시작")
    ax.set_ylim(0, 1)
    ax.set_xlabel("초")
    ax.set_title("밝기 시계열 — 포락선이 좁으면 자동 노출, 천천히 출렁이면 드리프트")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_png, dpi=110)
    log.info("그림 저장: %s", out_png)


# ────────────────────────── 공통 ──────────────────────────


def _use_korean_font(plt) -> None:
    """설치된 한글 글꼴 중 하나를 고른다. 없으면 경고만 끄고 진행한다."""
    import logging as _logging

    from matplotlib import font_manager

    _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    import warnings

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    width = max(len(k) for k, _ in rows)
    log.info("═══ %s ═══", title)
    for k, v in rows:
        log.info("  %-*s  %s", width, k, v)


def main() -> None:
    parser = argparse.ArgumentParser(description="실기기 녹음·녹화 파일 진단")
    parser.add_argument("path", type=Path, help="wav 또는 mp4 파일")
    parser.add_argument(
        "--out", type=Path, default=None, help="그림 저장 경로 (기본: 입력 파일 옆 .png)"
    )
    args = parser.parse_args()

    out_png = args.out or args.path.with_suffix(".png")
    suffix = args.path.suffix.lower()
    if suffix in (".wav", ".flac", ".m4a", ".mp3"):
        diagnose_wav(args.path, out_png)
    elif suffix in (".mp4", ".mov", ".avi", ".mkv"):
        diagnose_mp4(args.path, out_png)
    else:
        raise SystemExit(f"지원하지 않는 파일 형식: {suffix}")


if __name__ == "__main__":
    main()
