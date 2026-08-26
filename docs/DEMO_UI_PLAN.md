# 데모 UI 계획 (M5)

작성일: 2026-08-22. PLAN.md M5의 `demo/` 항목을 구체화한 문서.

---

## 1. 배경

팀이 참고용으로 SNS에서 찾은 영상 2개(우리 프로젝트가 아니라 다른 사람이 만든 비슷한
컨셉의 프로젝트):

- **DECIMEN — Fountain QR File Transfer**: 폰 브라우저 화면에 QR과 함께 통계 패널
  (Capture FPS, Decode FPS, Dropped, Goodput, Elapsed, Frames New/Dup/Red, Session,
  Payload)이 실시간으로 갱신되고, 전송이 끝나면 "Transfer Complete! — 365 KB in 2.6s
  (140.42 KB/s)" 배너가 뜬다.
- **SOUNDCHAT — Proximity Acoustic Radio**: 어두운 배경에 TX/RX 표시, 채널·주파수
  대역 표시, 실시간 파형, 로그 창, 타이핑 입력창으로 이뤄진 폰 앱 UI.

이 프로젝트의 `airgap/channels/screen_qr.py`, `airgap/channels/acoustic_fsk.py`는
이미 저 두 프로젝트와 같은 일(파운틴 QR 전송, 음향 채팅)을 한다 — **부족한 건 알고리즘이
아니라 "그 과정을 보여주는 화면"이다.** 그래서 이 문서는 물리층·측정 코드는 전혀
건드리지 않고, 순수하게 발표용 표시 레이어만 설계한다.

**주의 (CLAUDE.md 규칙 5와의 관계):** `demo/`는 원래 "실험 코드와 분리"돼 있는
영역이라 여기서 화려하게 꾸며도 채널 간 공정 비교 원칙과 충돌하지 않는다. 단,
`channels/*.py`(measurement 대상 코드) 자체는 한 글자도 안 바꾼다 — 통계 패널이
필요로 하는 값(fps, dropped 등)은 데모 쪽 래퍼가 채널을 감싸서 계산한다.

---

## 2. 설계 원칙 — 상태와 화면을 분리한다

세 가지 방식(rich 터미널, matplotlib 창, 웹 페이지)의 공통점은 **같은 숫자·로그를
다르게 그릴 뿐**이라는 것이다. 그래서 "상태를 채우는 부분"과 "상태를 그리는 부분"을
분리한다.

```
                     ┌─────────────────┐
  acoustic_chat.py → │                 │ → Tier 1: rich 패널 (터미널)
  screen_qr_demo.py → │     DemoHub     │ → Tier 2: matplotlib 파형 창
  (채널을 감싸는 코드) │  (상태 저장소)   │ → Tier 3: 웹 페이지 (SSE)
                     └─────────────────┘
```

이렇게 나누는 이유는 순수 소프트웨어 설계 취향이 아니라 **일정 위험 관리**다 — Tier 3가
시간이 없어 통째로 버려져도 Tier 1·2는 `DemoHub`를 그대로 재사용하므로 영향이 없다.

### `DemoHub` (개념 스케치 — 실제 필드는 구현 시점에 확정)

```python
@dataclass
class DemoStats:
    channel_name: str          # "acoustic_fsk" | "screen_qr"
    mode: str                  # "TX" | "RX" | "IDLE"
    capture_fps: float | None = None
    decode_fps: float | None = None
    dropped: int = 0
    goodput_kbps: float = 0.0
    elapsed_s: float = 0.0
    log_lines: list[str] = field(default_factory=list)   # 최근 N개만 보관
    last_chunk: np.ndarray | None = None                  # 파형 시각화용 (Tier 2)
```

`DemoHub`는 `update(**kwargs)` / `log(line)` 두 메서드만 있는 아주 얇은 클래스로 둔다
(옵저버 패턴 등 팀이 안 배운 개념을 끌어오지 않는다 — CLAUDE.md 규칙 1). 각 렌더러는
`DemoHub`의 최신 스냅샷을 주기적으로 읽어가기만 하면 된다.

---

## 3. 3단계 구현 계획

**전제: 이 단계는 M5(9/7~9/10)에서만 착수한다.** 지금(2026-08-22)은 M3(화면 채널
실기기 검증)가 완료 조건을 못 채운 상태라, PLAN.md의 "앞질러 가지 않는다" 원칙상
지금 코드를 쓰기 시작하지 않는다. M3~M4가 예정대로 끝나고 M5 윈도우에 들어가면
아래 순서로 진행한다.

### Tier 1 — `rich` 터미널 HUD (필수)

- **목표:** 콘솔 안에서 SOUNDCHAT과 비슷한 느낌(색깔 있는 패널, TX/RX 표시, 로그,
  실시간 갱신)을 낸다.
- **범위:** `demo/acoustic_chat.py`를 감싸는 얇은 레이어 추가(기존 `ChatReceiver`
  로직은 안 바꿈), `demo/screen_qr_demo.py` 신설 시 처음부터 이 위에서 작성.
  `rich.live.Live` + `rich.table.Table`이면 충분 — 새 개념 1개(rich의 Live 갱신 방식)만
  익히면 된다.
- **왜 1순위인가:** 새 스택이 필요 없고(터미널만 있으면 됨), 프로젝터·와이파이·방화벽
  문제에 영향받지 않는다. **발표장에서 다른 티어가 다 실패해도 이것만은 되게 한다.**
- **소요 추정:** 하루 미만 (기존 프로토타입에 얹는 수준).

### Tier 2 — matplotlib 실시간 파형 창 (권장)

- **목표:** SOUNDCHAT의 파형 바를 흉내낸 실시간 라인 플롯. `matplotlib`는 이미
  `requirements.txt`에 있고 팀이 M4 분석 그래프에서 어차피 쓸 라이브러리라 새로 배울
  게 거의 없다.
- **범위:** `DemoHub.last_chunk`를 `FuncAnimation`으로 주기 갱신. 텍스트 로그·통계
  패널까지 matplotlib으로 그리진 않는다(어색하고 번거로움 — Tier 1이 그 역할을 맡음).
  즉 Tier 2는 파형 시각화 **보조 창**이지 Tier 1의 대체가 아니다.
- **소요 추정:** 반나절.

### Tier 3 — 로컬 웹 페이지 (도전, 확장 과제급)

- **목표:** DECIMEN처럼 폰 브라우저로 통계 패널을 보여준다. 시각적으로 가장 인상적이지만
  팀이 한 번도 안 다뤄본 스택(Flask + 로컬 서버 + HTML/JS)이 새로 들어간다.
- **범위를 최소화하는 방법:** WebSocket 대신 **Server-Sent Events(SSE)** 사용 — 클라이언트가
  받기만 하면 되고 별도 라이브러리 없이 `EventSource`(브라우저 내장) + Flask 기본 기능만으로
  된다. 페이지 1장, 엔드포인트 2개(`/`, `/events`)로 제한.
- **왜 3순위·확장 과제급인가:** 실패 지점이 제일 많다 — 노트북·폰이 같은 네트워크에
  있어야 하고, 발표장 와이파이/방화벽이 막을 수 있다. PLAN.md의 확장 채널(E1~E3)과
  같은 원칙 적용: **M4가 끝나고 시간이 남을 때만 착수하고, 하나라도 M2~M5를 지연시키면
  즉시 중단한다.**
- **착수 조건:** Tier 1이 리허설까지 끝난 뒤에만. Tier 1 없이 Tier 3부터 만들지 않는다.

---

## 4. 적용 대상 정리

| 데모 | 기반 코드 | 상태 |
|---|---|---|
| 음향 채팅 | `demo/acoustic_chat.py` (기존, `ChatReceiver`는 안 바꿈) | Tier 1·2 래퍼 추가 예정 |
| 화면 QR 전송 | `demo/screen_qr_demo.py` (신설, `channels/screen_qr.py` 안 바꿈) | 통계 계산 래퍼부터 새로 작성 |

가시광 밝기 변조(`screen_flicker.py`)는 사람 눈에 깜빡임으로만 보이는 채널 특성상
"있어 보이는 UI"의 이득이 QR·음향보다 작다고 판단 — 이번 데모 UI 계획 대상에서 제외.
필요하면 M5에서 여유를 보고 추가 검토.

---

## 5. 완료 조건과의 관계

M5 완료 조건은 **"인터넷 없는 노트북 두 대로 시연 성공"**이지 UI의 화려함이 아니다.
이 문서의 티어 구분은 그 완료 조건을 지키면서 "있어 보이는" 수준을 최대한 올리기 위한
우선순위표일 뿐이다. Tier 1만 끝나도 M5 완료 조건은 충족된다 — Tier 2·3은 보너스.

PLAN.md 위험 요소 표에 "데모 UI 욕심이 M3~M4를 잠식" 행을 추가해뒀다: 대응은 동일한
원칙(필수/권장/도전 구분, 확장 과제와 같은 급으로 취급)이다.
