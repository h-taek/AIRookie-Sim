# AIRookie_Sim

> 소형 이족 동반 로봇 AI Rookie 의 시뮬레이터 앱이다. 맥에서 돌며, 앱 안에서 실제 온보드 SW 가 돈다.

[English](README.md) | 한국어

## 무엇을 해결하는가

자율 SW(호출, 찾기, 따라가기, 감정, 표현)는 실물이 나오기 전에 만들어야 했고, 이전 검증은 판정 결과를 밖에서 넣어 실제 파이프라인을 건너뛰었다. 이 앱은 실물에서 서보 · IMU · 카메라 · 마이크가 꽂히는 자리에 시뮬을 꽂는다. 온보드 SW 는 자기가 시뮬에서 도는지 실물에서 도는지 모르므로 검출 · 전사 · 판정이 실제로 돈다.

## 누구를 위한 것인가

- 하드웨어가 준비되기 전에 자율 SW 를 처음부터 끝까지 시험하려는 사람
- 학습한 걷기 정책을 두고 위젯을 붙인 MuJoCo 앱을 만들려는 사람

## 기능

- 학습과 같은 로봇 모델(서보 유격 포함)과 옮길 수 있는 3D 사람이 있는 MuJoCo 월드
- 로봇 카메라 시점과 구경꾼 시점을 나란히, 온보드 SW 조회 값을 실시간으로
- 검증 모드 둘: 추종(월드 렌더를 로봇 카메라로), 감정 표현(발화 얼굴 클립 또는 맥북 카메라)
- 소리 입력: 발화 WAV, 적으면 macOS TTS 가 읽는 글, 맥북 마이크
- 화면에서 정책 바꾸기, 젯슨 실물 진입점을 띄우고 그 카메라 · 조회를 같은 화면에 보이는 실물 모드

## 동작 방식

- 한 주기는 20 ms 다. `bridge.py` 가 월드 상태를 드라이버 계약의 센서 묶음으로 지어 온보드 SW 를 부르고, 지령을 월드에 돌려 쓴다.
- 온보드 SW 는 설치하지 않고 경로로 붙인다. `pyproject.toml` 이 `pythonpath = ["src", "../AIRookie_Runtime/src"]` 로 잡으므로 세 저장소를 나란히 둔다.

| 모듈 | 역할 |
|---|---|
| `world.py` | 물리 월드. 로봇과 사람, 초기 자세 |
| `bridge.py` | 월드 → 센서 묶음, 지령 → 월드 |
| `views.py` | 로봇 카메라 시점과 구경꾼 시점 렌더 |
| `sources.py` · `inject.py` | 영상 · 소리 입력원과 외부 주입 |
| `panels.py` · `app.py` | 조회 값 고르기와 화면 그리기(dearpygui) |

## 요구 사항과 사용법

- macOS, Python 3.11 이상, [uv](https://docs.astral.sh/uv/)
- [AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime) 과 [AIRookie_Playground](https://github.com/h-taek/AIRookie-Playground) 를 이 저장소 옆에 받아 둔다

```bash
uv venv
uv pip install mujoco dearpygui numpy onnxruntime openai-whisper pytest
uv pip install opencv-python-headless sounddevice      # 맥북 카메라 · 마이크, 발화 얼굴
```

사람 검출 모델은 온보드 저장소에서 한 번 만든다.

```bash
cd ../AIRookie_Runtime
uv run --no-project --with ultralytics --with onnx --with onnxslim python tools/export_detector.py
```

띄운다.

```bash
.venv/bin/python run.py                         # 기본 정책 v2_48
.venv/bin/python run.py --policy <정책.onnx>
.venv/bin/python run.py --real                  # 실물에 붙는다 (ssh 별칭 "jetson")
.venv/bin/python -m pytest -q                   # 장치를 안 건드리는 시험
.venv/bin/python -m pytest -q -m device         # 마이크 · 카메라 · TTS
```

## 자료

사람 메시와 감정 클립은 저장소에 없다. 메시는 `tools/prepare_person_mesh.py <메시.obj>` 로 맞춘다. 감정 클립은 AI Hub "한국인 감정인식을 위한 복합영상"에서 자른 것이고 재배포가 허용되지 않으므로, 각자 AI Hub 에서 받아 `assets/emotion_video/` 에 둔다.

## 관련 저장소

| 저장소 | 역할 |
|---|---|
| [AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime) | 이 앱 안에서 도는 온보드 SW |
| [AIRookie_Playground](https://github.com/h-taek/AIRookie-Playground) | 걷기 정책과 로봇 모델 |

## 라이선스

[MIT License](LICENSE) 를 따른다.
