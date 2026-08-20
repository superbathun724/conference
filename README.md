# Air-Gap

인터넷을 쓰지 않고 정보를 전달하는 여러 물리 채널을 하나의 소프트웨어 구조 위에 구현하고,
전송 성능과 신호 선별의 한계를 공정하게 비교하는 프로젝트.

> **연구 질문**
> 인터넷을 사용하지 않고 정보를 전달할 때,
> 채널의 물리적 성질은 전송 한계와 신호 선별의 한계를 어떻게 결정하는가?

2026 지식 나눔 학술제 '광장' 출품작 · 팀 에어갭 (3인)

---

## 두 축

| | 묻는 것 | 담당 |
|---|---|---|
| **축 A · 전송** | 정보를 얼마나 빠르게 보낼 수 있는가 | 각자 자기 채널 |
| **축 B · 선별** | 원하는 신호만 어떻게 골라낼 것인가 | 각자 자기 채널 |

두 축은 같은 물리적 제약(시간·주파수 맞바꿈)에 묶여 있다. 이것을 실측으로 보이는 것이 이 프로젝트의 목표다.

## 채널

| 채널 | 매질 · 파동 | 부호화 축 | 지위 |
|---|---|---|---|
| 음향 FSK | 공기 · 종파 | 시간 | 필수 (최소 성공선) |
| 화면 밝기 변조 | 가시광 · 전자기파 | 시간 | 필수 (QR의 대조군) |
| 동적 QR | 가시광 · 전자기파 | 시간 + 공간 | 필수 |
| 블루투스 | 전파 · 전자기파 | — | 비교 기준선 (구현 안 함) |
| 근초음파 · 진동 · 자기장 | 공기 · 고체 · 근접장 | 시간 | 확장 과제 |

## 문서

| 파일 | 내용 | 언제 읽나 |
|---|---|---|
| `CLAUDE.md` | 프로젝트 상수, 절대 규칙, 환경 설정 | 세션마다 |
| `PLAN.md` | 단계별 작업 순서와 완료 조건 | 작업 시작 전 |
| `docs/ARCHITECTURE.md` | 계층 구조, 인터페이스, 확장 방법 | 코드를 건드리기 전 |
| `docs/EXPERIMENTS.md` | 측정 규약, 데이터 스키마, 변인 설계 | 측정하기 전 |
| `docs/THEORY.md` | 물리·생물 이론 배경, 자주 틀리는 것 | 보고서 쓸 때 |
| `FINDINGS.md` | 시도와 결과의 기록 | 계속 |

## 시작하기 (Windows)

```powershell
cd ~\Desktop\project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m sounddevice          # 오디오 장치 인덱스 확인
pytest                         # 루프백 왕복 테스트

# M1 실측 (노트북 두 대, 또는 한 대 + 외부 마이크를 1m 거리에 두고)
# 수신측 먼저 실행:
python -m airgap.experiments.acoustic_recv --trials 10 --distance-cm 100
# 송신측:
python -m airgap.experiments.acoustic_send --trials 10
```

`demo/` 아래 발표용 스크립트는 M5에서 만든다. 지금은 `experiments/acoustic_send.py` /
`acoustic_recv.py`가 음향 채널의 실기기 검증용이다.

### 노트북 오디오 장치가 없을 때 — 안드로이드 두 대로 대체

노트북 오디오 드라이버가 고장났거나 마이크/스피커가 없으면, 라이브 재생·녹음 대신
파일을 거쳐 검증할 수 있다. `sounddevice`(스피커/마이크)를 전혀 쓰지 않으므로
노트북 오디오 장치 상태와 무관하게 동작한다.

```powershell
# 1. 노트북에서 시행별 WAV 생성 (스피커로 재생하지 않고 파일로 저장)
python -m airgap.experiments.acoustic_send --trials 10 --out-dir out\send

# 2. out\send의 trial_01.wav ... 를 폰 A로 옮겨 순서대로 재생
#    폰 B를 1m 거리에 두고 각 재생마다 녹음 (반드시 WAV로 저장 — m4a/aac는
#    soundfile이 못 읽는다. 안 되면 ffmpeg -i in.m4a -ar 44100 -ac 1 out.wav로 변환)
#    녹음 파일을 trial_01.wav ... 이름으로 in\recv에 모아 노트북으로 옮긴다

# 3. 노트북에서 판정
python -m airgap.experiments.acoustic_recv --in-dir in\recv --distance-cm 100 `
    --expected-payload "AIRGAP 20자 테스트문자열"
```

노트북 드라이버가 고쳐지면 `--out-dir`/`--in-dir` 없이 위쪽 라이브 경로를 그대로
다시 돌려 성공률을 비교한다 (자세한 배경은 `FINDINGS.md` 2026-08-20 항목).

## 일정

| 날짜 | 마감 |
|---|---|
| 2026. 8. 14. | 활동 계획서 (완료) |
| 2026. 8. 28. | 중간 보고서 |
| 2026. 9. 10. | 최종 보고서 및 발표 |
