"""시뮬레이터 앱.

**시계는 앱이 쥐고 실제 시간에 맞춘다** (사양 1-2). 온보드 SW 는 스스로
루프를 돌지 아니하며, 이 파일이 벽시계 20 ms 마다 한 주기씩 불러 준다.
일시정지와 한 주기 진행은 없다. 되돌리는 것은 초기화 하나다.

한 주기가 늦으면 세상은 계속 흐른다. 늦은 만큼 물리를 밀어 따라잡고
온보드 SW 는 최신 센서로 한 번만 부른다.

`Sim` 이 한 주기를 맡고 `App` 이 창을 맡는다. 가른 이유는 `Sim` 이 창
없이 시험되기 때문이다 — 화면은 눈으로 봐야 하지만 고리는 그럴 필요가 없다.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import sources
from bridge import Bridge
from devices import pipe_camera, pipe_mic
from expression import PRESET_NAMES
from inject import Injector, 모드_감정, 모드_추종, 모드들
from launcher import (Launcher, asr_url, start_asr_server, start_emotion_server, stop_asr_server,
                      맥_감정_포트, 맥_전사_포트)
from mirror import Mirror
from panels import History, box_rects, sections, summary_rows, target_id
from real_link import Real
import runtime as _runtime
from runtime import Command, ManualInput, Runtime, apply_app_command, load_autonomy, load_home_pose
from senses import Senses
from views import Views
from world import World

# 제어 주기다. 온보드 SW 가 50 Hz 로 돈다.
제어_주기_s = 0.02

# 카메라는 자기 박자로 돈다. 4K 를 제어 주기마다 렌더하면 예산을 넘는다.
#
# **제어 주기의 약수여야 한다.** 프레임을 주기 경계에서만 찍으므로 50 Hz
# 격자에 안 떨어지는 값을 적으면 적은 대로 나오지 아니한다 -- 30 을 적으면
# 실제로는 25 가 나온다. 잠정값이며 5-4 에서 한 주기 시간을 보고 조정한다.
카메라_fps = 25.0

# 몇 주기에 한 번 찍는가. 시각을 빼서 견주면 0.02 를 더한 오차가 쌓여
# 프레임이 조용히 빠진다 -- 100주기에 50장이 아니라 48장이 나왔다.
카메라_주기_수 = round(1.0 / (카메라_fps * 제어_주기_s))

# 한글 글꼴이다. dearpygui 의 기본 글꼴에는 한글 글리프가 없어 없는 글자를
# 물음표로 그린다. 창 제목은 macOS 가 그리므로 멀쩡해 보이는데 위젯 안 글자만
# 깨진다 -- 인코딩 문제로 보이지만 글꼴 문제다.
#
# 앞의 것부터 찾아 쓰고, 하나도 없으면 글꼴을 물리지 아니한다. 맥이 아닌
# 곳에서도 앱이 뜨기는 해야 한다.
한글_글꼴_후보 = (
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)
글꼴_크기 = 16

# 화면을 그리는 박자다. **창의 수직 동기에 고리를 묶지 아니한다** -- 묶으면
# 한 바퀴가 화면 주사율에 끌려가 60 Hz 화면과 50 Hz 제어가 엇박이 나고,
# 그 엇박이 빠진 주기로 잡힌다. 고리는 제어 박자로 돌고 그림은 이 박자로 늦춘다.
화면_fps = 30.0

# 글자를 갱신하는 주기다. 그림은 매 프레임 갱신하되 숫자는 늦춘다 --
# 60 Hz 로 바뀌는 숫자는 읽을 수가 없다. 시뮬 시각이 아니라 벽시계로 센다.
글자_갱신_주기_s = 0.3

# 세 자리의 폭이다. 좌우는 고정이고 가운데가 남는 만큼 가져간다.
# 창은 16:9 로 열되, 화면에서 쓸 수 있는 자리보다 크면 그 자리에 맞춘다 (`쓸_자리`).
창_폭 = 1920
창_높이 = 1080
# macOS 창 제목 막대의 세로(점)다. 창 크기와 자리는 제목 막대를 뺀 안쪽 기준이라 따로 뺀다.
제목_막대 = 28
좌측_폭 = 260
# 좌측 밀대 · 숫자 칸의 폭이다. 오른쪽에 붙는 이름 몫을 남긴다.
칸_폭 = -90
# 좌측 소제목 · 상태 글자색이다.
소제목_색 = (150, 160, 175)
상태_색 = (140, 140, 140)
우측_폭 = 440
# 실물의 실행 로그 패널 세로다. 창 폭 전체를 쓰는 아래 패널이다 (사양 4-4).
로그_높이 = 220
# 창 안쪽 여백과 위젯 사이 세로 간격이다 (dearpygui 기본값).
창_안쪽 = 8
# 라벨 한 줄의 세로다. 창이 서기 전에는 잴 수 없어 이 값으로 시작한다.
라벨_세로 = 22
# 위·가운데·아래에 똑같이 두는 간격이다.
간격_목표 = 12
# dearpygui 가 위젯 사이에 넣는 간격이다. 실측으로 얻었다.
위젯_사이 = 4

# 사람을 키로 움직이는 속도다. 잠정값이며 걷는 속도쯤으로 잡았다.
# 월드 조작은 온보드 SW 를 거치지 아니하므로(사양 3-4) 제어 주기가 아니라
# 창 박자로 움직인다.
사람_속도_mps = 1.2
사람_회전_dps = 90.0

# 수동 조작 토글 버튼. 구경꾼 그림 우측 상단에서 이만큼 안쪽에 둔다 (px).
토글_폭 = 110
토글_여백 = 8

# 걷기 정책이다. 온보드 SW 를 만들 때 건넨다 (계약 1-2). 학습 저장소의 것을
# 쓰며 이 저장소는 담지 아니한다.
_플레이그라운드 = Path(__file__).resolve().parents[2] / "AIRookie_Playground"
정책_폴더 = _플레이그라운드 / "policies"

# 정책은 외장별 하위 폴더(`v1/` · `v2/`)에 있고 이름은 `v{외장}_{회차}_{스텝}.onnx` 다 (Playground `policies/README.md`).
# 회차 뒤에 판 글자(`10B`)가, 스텝 자리에 중간 체크포인트(`mid64M`)가 올 수 있다.
_정책_이름 = re.compile(r"v(\d+)_(\d+)[A-Za-z]*_(\w+)\.onnx")


# 시뮬 · 실물이 처음 고르는 걷기 정책이다. 실물에서 지금 쓰는 정책이다.
기본_정책_이름 = "v2_48_300M.onnx"


def 기본으로_고를_정책(폴더: Path = 정책_폴더) -> Path:
    """`기본_정책_이름` 의 정책 파일이다. 외장별 하위 폴더까지 찾는다."""
    찾음 = sorted(폴더.rglob(기본_정책_이름))
    if not 찾음:
        raise FileNotFoundError(f"{폴더} 에 {기본_정책_이름} 이 없다")
    return 찾음[0]


기본_정책 = 기본으로_고를_정책()
# 정책과 짝인 시범 동작이다. 관측의 보행 위상이 이 파일의 주기로 돈다.
시범_동작 = _플레이그라운드 / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"


def 쓸_자리() -> tuple[int, int, int, int] | None:
    """주 화면에서 메뉴 막대와 독을 뺀 자리 (x, y, 폭, 높이). 못 알면 없다.

    **픽셀이 아니라 점(point)으로 센다.** 창 크기도 점이다. 레티나는 한 점이 두 픽셀이라
    픽셀로 재면 창이 화면의 두 배까지 커진다. y 는 화면 위에서 잰다.
    """
    if sys.platform != "darwin":
        return None
    글 = (
        'ObjC.import("AppKit"); var s = $.NSScreen.mainScreen; var v = s.visibleFrame, f = s.frame;'
        "[v.origin.x, f.size.height - v.origin.y - v.size.height, v.size.width, v.size.height].join(' ')"
    )
    try:
        r = subprocess.run(["osascript", "-l", "JavaScript", "-e", 글], capture_output=True, text=True, timeout=3)
        x, y, 폭, 높이 = (int(float(v)) for v in r.stdout.split())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return x, y, 폭, 높이


def 새_런타임(policy: Path | str = 기본_정책, senses: Senses | None = None, safety_on: bool = True):
    """온보드 SW 를 새로 만든다. 앱과 시험이 같은 길로 만든다.

    `senses` 를 건네지 않으면 장치 없는 지각 · 감정을 스레드로 스스로 띄운다 (시험의 길).
    `safety_on` 은 안전 계층이다. 시뮬은 켜고 시작한다 (Runtime 자율 사양 7-1).
    """
    return Runtime(policy, 시범_동작, safety_on=safety_on, senses=senses)


def 정책들() -> list[Path]:
    """고를 수 있는 정책 파일이다. 외장별 하위 폴더까지 이름 차례로 낸다."""
    return sorted(p for p in 정책_폴더.rglob("*.onnx") if _정책_이름.fullmatch(p.name))


# 감정 표현 검증에 쓸 영상 자료다. AI Hub 원본에서 자른 클립이며
# 저장소에 담지 않는다. 없어도 앱은 선다.
감정_영상_폴더 = Path(__file__).resolve().parents[1] / "assets" / "emotion_video"

# 화면에 적는 이름이다. 코드 · 기록의 이름은 그대로 둔다.
표현_이름 = {
    "call_heard": "호출 들음",
    "listening": "듣는 중",
    "heard": "들었음",
    "react_positive": "긍정 반응",
    "react_negative": "부정 반응",
    "react_neutral": "중립 반응",
}
모드_이름 = {모드_추종: "3D 장면", 모드_감정: "감정 영상"}

# 실물 디버깅 도구 (사양 4-5). 효과음 파일은 Runtime 저장소의 것이다.
_런타임_저장소 = Path(_runtime.__file__).resolve().parents[1]
효과음_폴더 = _런타임_저장소 / "assets" / "sounds"

# 지난번에 쓴 입력값이다 (사양 4-6). 창을 닫을 때 적고 열 때 되살린다. 맥마다 따로다.
입력_기록 = Path.home() / ".config" / "airookie_sim" / "inputs.json"
# 고르고 적는 입력이다. 사람 위치 · 전압 · 온도는 월드 상태라 넣지 않는다.
기억할_입력 = (
    "link_route", "asr", "emotion",
    "run_arm", "run_safety", "run_no_servo", "run_servo_speed",
    "speed_vx_max", "speed_vx_back_max", "speed_vy_max", "speed_wz_max",
    "real_mic_source", "real_sound", "real_volume",
    "verify_mode", "video_clip", "tts_voice", "tts_text", "mic_device", "cam_index",
)


def 지난_입력() -> dict:
    try:
        return json.loads(입력_기록.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class Sim:
    """한 주기를 처리한다. 창을 알지 못한다."""

    def __init__(self, policy: Path | str = 기본_정책, container: str = "process", asr: str = "",
                 asr_url: str = "", safety: bool = True, emotion: str = "", emotion_url: str = "") -> None:
        self.policy = Path(policy)
        self.safety = bool(safety)
        # 실행 중인가 (`launch` · `stop`). 멈추면 온보드 SW 를 부르지 않고 홈 자세를 쥔다 -- 실물 Ctrl-C 와 같다.
        self.running = True
        # 실물 `--no-servo` 흉내. 로봇을 홈 키프레임에 묶고 센서도 홈 값이다 (`World.hold_home`).
        self.no_servo = False
        # 모드 전이 · 전사문 · 감정 · 표현 · 효과음의 최근 기록. 실물과 같은 꼴이다 (`panels.History`).
        self.history = History()
        # 앱이 보낸 수동 속도 · 볼륨. 온보드 SW 를 새로 만들어도 다시 건다 (`_앱_명령을_다시_건다`).
        self._앱_명령: dict[str, dict] = {}
        self.world = World()
        self.bridge = Bridge(self.world)
        # 지각 · 감정은 실물과 똑같이 제 프로세스로 한 번 띄운다. 꽂는 장치만 다르다 —
        # 로봇 카메라 렌더와 흘려 넣는 소리를 놓는 입구다 (Runtime spec/05 4장).
        # 초기화해도 다시 띄우지 않는다. 모델을 다시 올리지 않기 위해서다.
        카메라, self.camera_in = pipe_camera()
        마이크, self.mic_in = pipe_mic()
        # 전사 백엔드는 실물의 `real/run.py --asr · --asr-url` 과 같은 칸을 덮어쓴다. 비우면 설정 그대로다.
        설정 = load_autonomy()
        if asr:
            설정["perception"]["asr_backend"] = asr
        if asr_url:
            설정["perception"]["asr_mac_url"] = asr_url
        self.asr = 설정["perception"]["asr_backend"]
        # 감정 추론 자리도 실물의 `--emotion · --emotion-url` 과 같은 칸을 덮어쓴다.
        if emotion:
            설정["emotion"]["backend"] = emotion
        if emotion_url:
            설정["emotion"]["mac_url"] = emotion_url
        self.emotion = 설정["emotion"]["backend"]
        # 앱이 고른 (전사, 맥 주소, 감정, 맥 주소). 바뀌면 앱이 지각 · 감정을 새로 띄운다 (`App._시뮬을_띄운다`).
        self.asr_choice = (asr, asr_url, emotion, emotion_url)
        # 효과음은 맥 스피커로 낸다 (Runtime spec/04 11장).
        self.senses = Senses(camera=카메라, mic=마이크, container=container, speaker=["afplay"], settings=설정)
        self.runtime = 새_런타임(self.policy, self.senses, self.safety)
        # 실물과 같이 모델이 다 선 뒤에 시계를 돌린다. 초기화 때는 모델이 그대로라 기다리지 않는다.
        self.runtime.wait_ready(report=lambda 남은: print(f"모델 대기 — 남은 것 {', '.join(남은)}" if 남은 else "모델 준비 끝"))
        # 힘을 놓은 귀를 처음에 둘 자리다. 온보드 SW 의 쉬는 귀 각도를 그대로 쓴다.
        표현_설정 = Path(_runtime.__file__).parent / "config/expressions.json"
        쉬는_귀 = json.loads(표현_설정.read_text(encoding="utf-8"))["rest"]["ear"]
        self.views = Views(self.world.model, ear_rest_deg=쉬는_귀)
        self.injector = Injector(self.bridge)
        self.joint_names, self._홈 = load_home_pose()

        # 수동 조종 (Runtime 자율 사양 8장). 실물의 조종기처럼 센서 묶음에 실어 보낸다.
        # 켜져 있는 동안만 싣고, 꺼지면 온보드 SW 가 자율로 돈다.
        self.manual = False
        self._조종 = ManualInput()

        # 모드를 바꾼 직후에 앞 모드의 마지막 그림이 남지 않게 덮을 것이다.
        너비, 높이 = self.views.robot_size
        self._검은_프레임 = np.zeros((높이, 너비, 3), dtype=np.uint8)

        self._시계를_0으로()

    def _시계를_0으로(self) -> None:
        self.t = 0.0
        self.cycles = 0
        # 늦어서 온보드 SW 를 못 부른 주기 수다. 물리는 그 주기에도 흘렀다.
        self.missed = 0
        self.frame: np.ndarray | None = None
        self.cycle_ms = 0.0
        # 마지막으로 찍은 카메라 박자다. 주기를 건너뛰어도 박자를 넘겼으면 찍는다.
        self._찍은_박자 = -1

    def due(self, elapsed_s: float) -> int:
        """벽시계로 `elapsed_s` 가 흘렀을 때 밀어야 할 주기 수다.

        k 번째 주기는 벽시계가 k·20 ms 를 넘긴 뒤에 돈다. 0 이면 아직 때가 아니다.
        """
        # 0.06 / 0.02 가 2.9999... 로 떨어져 한 주기씩 늦게 잡히지 않게 한다.
        지난_주기 = math.floor(elapsed_s / 제어_주기_s + 1e-9) + 1
        return max(0, 지난_주기 - self.cycles)

    def cycle(self, ticks: int = 1) -> None:
        """주기를 민다. 센서를 모으고, 온보드 SW 를 부르고, 물리를 민다.

        `ticks` 가 1 보다 크면 늦은 것이다. **세상은 계속 흐른다** (사양 1-2) --
        빠진 주기만큼 물리를 먼저 밀고, 온보드 SW 는 최신 센서로 한 번만 부른다.
        빠진 주기에는 지령이 없으므로 서보는 직전 지령을 쥔다.
        """
        if ticks < 1:
            raise ValueError(f"ticks 는 1 이상이어야 한다: {ticks}")
        시작 = time.perf_counter()

        빠진 = ticks - 1
        if 빠진:
            self.world.step(빠진 * 제어_주기_s)
            self.cycles += 빠진
            self.missed += 빠진
            self.t = self.cycles * 제어_주기_s

        frame = self._capture()
        if frame is not None:
            self.camera_in.put(self.t, frame)
        # 마이크는 빠진 주기에도 들었다. 그 소리를 버리지 아니한다.
        소리 = self.injector.audio_block(ticks * 제어_주기_s)
        if 소리 is None:
            # **로봇 마이크는 늘 듣는다.** 틀 것이 없으면 무음을 넣는다. 비워
            # 두면 발화 뒤의 무음이 오지 않아 발화가 영영 닫히지 아니한다.
            소리 = np.zeros(round(sources.기본_표본율 * ticks * 제어_주기_s), dtype=np.int16)
        self.mic_in.put(self.t, 소리)
        sensors = self.bridge.sensors(t=self.t, manual=self._조종 if self.manual else None)
        if self.running:
            self.bridge.apply(self.runtime.step(sensors))
        else:
            self.bridge.apply(Command(angle=self._홈))
        self.world.step(제어_주기_s)
        if self.no_servo:
            self.world.hold_home()
        self.cycles += 1
        self.t = self.cycles * 제어_주기_s

        self.cycle_ms = (time.perf_counter() - 시작) * 1000.0

    def reset(self) -> None:
        """처음으로 되돌린다 (사양 1-2).

        월드, 온보드 SW, 시계, 틀던 입력원, 전압·온도를 되돌린다.
        검증 모드는 그대로 둔다. 온보드 SW 는 새로 만든다 -- 안에 쌓인 상태를
        하나하나 지우는 것보다 확실하다. 지각 · 감정은 다시 띄우지 않고 비운다.
        """
        self.injector.stop()
        self.injector.clear()
        self.world.reset()
        self.runtime.close()
        self.senses.reset()
        self.runtime = 새_런타임(self.policy, self.senses, self.safety)
        self._앱_명령을_다시_건다()
        self._시계를_0으로()
        self.frame = self._검은_프레임

    def set_manual(self, on: bool) -> None:
        """수동 조종과 자율(추종)을 오간다. 바꿀 때 조종기 입력은 놓은 상태로 둔다."""
        self.manual = bool(on)
        self._조종 = ManualInput()

    def drive(
        self, forward: float, lateral: float, turn: float,
        head_pitch: float = 0.0, head_yaw: float = 0.0, head_home: bool = False,
        ear: tuple[bool, bool] = (False, False),
    ) -> None:
        """조종기 입력을 정한다. 걷기 셋이 다 0 이면 로봇이 선다 (자율 사양 8장)."""
        self._조종 = ManualInput(
            forward=forward, lateral=lateral, turn=turn,
            head_pitch=head_pitch, head_yaw=head_yaw, head_home=head_home, ear=ear,
        )

    def set_policy(self, policy: Path | str) -> None:
        """걷기 정책을 바꾼다. 온보드 SW 를 새로 만들어 처음부터 돈다 (사양 1-3).

        새 정책을 먼저 연다. 못 열면 앞 정책이 그대로 남는다.
        """
        새것 = 새_런타임(policy, self.senses, self.safety)
        self.reset()
        self.runtime.close()
        self.policy = Path(policy)
        self.runtime = 새것
        self._앱_명령을_다시_건다()

    def send_command(self, command: dict) -> bool:
        """앱의 디버깅 · 수동 속도 명령을 실물과 같은 입구(`runtime.apply_app_command`)로 넘긴다 (사양 4-4 · 4-5).

        못 따르면 까닭을 찍고 거짓이다. 수동 속도 · 볼륨은 기억해 두었다가 온보드 SW 를 새로 만들면 다시 건다.
        """
        try:
            if command.get("kind") == "manual":
                self.set_manual(command["on"])
                return True
            apply_app_command(self.runtime, command)
        except Exception as 오류:  # noqa: BLE001 -- 앱 단추 하나가 고리를 세우지 않게 한다
            print(f"앱 명령을 못 따랐다 — {command}: {오류}")
            return False
        if command["kind"] in ("manual_speed", "volume"):
            self._앱_명령[command["kind"]] = command
        return True

    def _앱_명령을_다시_건다(self) -> None:
        for 명령 in self._앱_명령.values():
            apply_app_command(self.runtime, 명령)

    def launch(self, policy: Path | str, arm: bool = True, safety: bool = True, no_servo: bool = False,
               servo_speed: int = 0) -> None:
        """실물 진입점의 실행 인자(--arm · --safety · --no-servo · --servo-speed)를 걸고 처음부터 돈다."""
        self.safety = bool(safety)
        if Path(policy) != self.policy:
            self.set_policy(policy)
        else:
            self.reset()
        self.bridge.set_armed(arm)
        self.no_servo = bool(no_servo)
        self.set_servo_speed(servo_speed)
        self.running = True

    def stop(self) -> None:
        """실물 Ctrl-C 와 같다. 온보드 SW 를 멈추고 홈 자세 목표를 쥔다. 다시 돌리려면 `launch` 다."""
        self.running = False

    def set_servo_speed(self, speed: int) -> None:
        """서보 속도 상한(step/s)을 건다. 0 이면 없다. 실물 `--servo-speed` 흉내다 (`bridge.Bridge.apply`)."""
        self.bridge.servo_speed = max(0, int(speed))

    def _capture(self) -> np.ndarray | None:
        """새 프레임이 있을 때만 낸다. 없는 주기에는 칸이 빈다 (계약 2-3).

        프레임의 출처가 모드에 따라 갈린다 (사양 3-3). 감정 표현 모드에서는
        **월드를 렌더하지 아니한다** -- 로봇 카메라에 들어가는 것이 주입한
        그림이므로 렌더한 것은 그대로 버릴 프레임이 된다.

        **나머지가 0 인 주기를 기다리지 아니한다.** 늦어서 주기를 건너뛰면
        그 주기가 빠질 수 있다. 카메라 박자를 넘겼는지로 본다.
        """
        박자 = self.cycles // 카메라_주기_수
        if 박자 == self._찍은_박자:
            return None
        self._찍은_박자 = 박자

        if self.injector.mode == 모드_감정:
            frame = self.injector.frame(self.t)
        else:
            frame = self.views.robot_frame(self.world.data, self.bridge.ear, self.bridge.eye)

        if frame is not None:
            self.frame = frame
        return frame

    def set_mode(self, mode: str) -> None:
        """검증 모드를 바꾸고 로봇 카메라를 비운다.

        비우지 않으면 바꾼 직후에 앞 모드의 마지막 그림이 그대로 남아, 아직
        새 프레임이 안 온 것인지 모드가 안 바뀐 것인지 가릴 수 없다.
        """
        self.injector.set_mode(mode)
        self.frame = self._검은_프레임

    def trigger_expression(self, name: str) -> None:
        """표현 프리셋 하나를 온보드 재생기에 직접 건다. **검증 전용이다** (사양 3-5).

        파이프라인은 있으나 걷기 정책의 결함으로 표현까지 닿지 못할 때 표현만 따로
        본다. 머리는 걷기 정책을 거쳐 나간다.
        """
        self.runtime.trigger_expression(name)

    def status(self):
        """우측 패널이 그릴 것이다. 최근 기록도 이때 채운다."""
        조회 = self.runtime.snapshot()
        if 조회 is not None:
            self.history.update(조회)
        return sections(조회, self.joint_names)

    def summary(self):
        """우측 패널 맨 위 요약이다."""
        return summary_rows(self.runtime.snapshot())

    def close(self) -> None:
        # 마이크와 카메라를 먼저 놓는다. 창이 닫힌 뒤에도 잡고 있으면
        # 다음에 앱을 띄울 때 장치를 못 연다.
        self.injector.stop()
        self.views.close()
        self.runtime.close()
        self.senses.close()
        self.camera_in.close()
        self.mic_in.close()


# 우측 맨 위 요약의 줄 이름과 차례다 (`panels.summary_rows`).
요약_칸 = ("모드", "대상", "안전", "전사문", "감정", "표현", "서보 전압 최저", "서보 온도 최고")


class App:
    """창과 네 자리. 수동 렌더 루프로 돈다.

    좌측은 월드 조작과 외부 주입, 중앙 위는 로봇 카메라 시점, 중앙 아래는
    구경꾼 시점, 우측은 조회다 (사양 4-1).

    **터미널로 입력받지 아니한다** (사양 4-2). 누르는 것은 전부 이 창에서 받는다.
    """

    def __init__(
        self, sim: Sim | None = None, real: bool = False, host: str = "jetson", launcher: Launcher | None = None,
    ) -> None:
        # 시뮬과 실물을 한 앱에서 오간다 (사양 4-4). 지금 보이는 쪽이 `self.sim` 이다 — 실물이면
        # `Sim` 자리에 `real_link.Real` 이 든다. 실물은 보기만 하고, 젯슨 실행은 `launcher` 가 맡는다.
        self.host = host
        건넨_실물 = bool(getattr(sim, "real", False))
        self.real = bool(real or 건넨_실물)
        self._시뮬: Sim | None = None if 건넨_실물 else sim
        self._실물: Real | None = sim if 건넨_실물 else None
        self.launcher: Launcher | None = None
        # 버튼이 고른 쪽이다. 바꾸는 일은 고리가 한다 (`_소스를_바꾼다`).
        self._바꿀_쪽: str | None = None
        # 젯슨에 붙는 경로. 고른 쪽으로만 붙는다 (`launcher.Launcher.route`).
        self._경로 = "같은 망"
        # 정책 칸은 시뮬과 실물이 같이 쓰되 고른 것은 쪽마다 따로 든다. 보이지 않는 쪽 것이다.
        self._실물_정책 = ""
        self._시뮬_정책: str | None = None
        # 앱이 띄운 맥 전사 · 감정 서버 (`launcher.start_asr_server` · `start_emotion_server`). 이미 떠 있었으면 없다.
        self._전사_서버 = None
        self._감정_서버 = None
        if self.real:
            if self._실물 is None:
                self._실물 = Real(host=None)
            self.launcher = launcher if launcher is not None else Launcher(
                host, tunnel_port=self._실물.port, route=self._경로)
        elif self._시뮬 is None:
            self._시뮬 = Sim()
        # 시계를 센 기준 시각이다. 첫 바퀴와 초기화 뒤에 잡는다.
        self._시작: float | None = None
        # 딴 실이 다 만든 TTS 소리다. 고리가 받아서 튼다 -- 주입기는 고리의
        # 것이라 딴 실이 직접 만지지 아니한다.
        self._만든_소리: list = []
        self._소리_자물쇠 = threading.Lock()

        # 받아 둔 영상 클립들이다. 자료가 없으면 빈 목록이다.
        self._영상들 = sources.video_clips(감정_영상_폴더)
        self._고른_영상 = 0

        self._시계 = time.perf_counter
        # 음의 무한대로 시작해야 첫 프레임이 걸러지지 아니한다. 0 으로 두면
        # 창을 열자마자 갱신 주기만큼 빈 패널이 보인다.
        self._마지막_글자 = float("-inf")
        self._마지막_화면 = float("-inf")
        # 글자를 늦추면 그 사이 한 주기 값이 여럿 지나간다. 직전 하나만
        # 띄우면 어느 것이 걸렸느냐에 따라 숫자가 널뛴다. 모아서 낸다.
        self._주기_시간: list[float] = []
        # 표를 다시 만들지 말지 가르는 값이다. 제목·머리줄·줄 수·열 수가 같으면
        # 위젯을 그대로 두고 칸만 갈아 끼운다.
        self._표_모양: tuple | None = None

        # 마우스가 구경꾼 그림 위에 있는가. 창이 서면 진짜 판정으로 갈아 끼운다.
        # 콜백 안에서 dearpygui 를 부르면 창 없이 시험할 수 없다.
        self._그림_위인가 = lambda: False
        # 그림 한 장의 세로다. 그린 뒤 실제 자리를 재서 다듬는다.
        self._그림_높이 = 400.0

        # 눌린 키를 묻는 것이다. 창이 서면 진짜 판정으로 갈아 끼운다.
        self._키가_눌렸나 = lambda 이름: False
        self._키_앞_시각: float | None = None
        self._토글_자리: list[int] | None = None
        self._조작_표시: tuple[str, str] | None = None
        # 실물에 고른 볼륨 · 수동 속도. 송출에 새로 붙을 때마다 다시 보낸다.
        self._앞_전송: str | None = None
        self._볼륨 = 100
        self._수동_속도값 = {키: float(load_autonomy()["manual"][키]) for 키 in ("vx_max", "vx_back_max", "vy_max", "wz_max")}
        # 로봇 시점을 복제해 띄운 딴 창이다 (사양 4-7). 닫혀 있으면 없다.
        self._거울: Mirror | None = None
        self._거울_바꿈 = False
        # 구경꾼 시점을 끌고 있는가. 그림 위에서 누른 것만 받는다.
        self._구경꾼_잡음 = False
        # dearpygui 의 끌기 이벤트는 누른 지점부터의 **누적** 이동량을 준다.
        # 그대로 더하면 끄는 동안 회전이 점점 빨라진다. 차이만 쓴다.
        self._끌기_앞 = (0.0, 0.0)

        # 텍스처가 물고 있는 버퍼다. 매 주기 새 배열을 만들지 않고 여기에 덮어쓴다.
        너비, 높이 = self.sim.views.robot_size
        구_너비, 구_높이 = self.sim.views.spectator_size
        self._로봇_버퍼 = self._빈_텍스처(너비, 높이)
        self._구경꾼_버퍼 = self._빈_텍스처(구_너비, 구_높이)

    # --- 창을 여는 쪽이 부른다 ---

    def run(self) -> None:
        """창을 띄우고 수동 루프를 돈다. 창이 닫히면 돌아온다."""
        import dearpygui.dearpygui as dpg

        # 문맥 생성이 항상 먼저다. 그 전에 다른 API 를 부르면 세그폴트가 난다.
        dpg.create_context()
        # 맥 Qwen 전사 서버와 em7 감정 서버는 앱과 같은 맥에서 돈다. 앱이 켜질 때 띄우고 꺼질 때 내린다.
        self._전사_서버 = start_asr_server()
        self._감정_서버 = start_emotion_server()
        try:
            self._화면을_짓는다(dpg)
            self._입력을_되살린다(dpg, 지난_입력())
            자리 = 쓸_자리()
            x, y, 폭, 높이 = 자리 if 자리 is not None else (0, 0, 창_폭, 창_높이 + 제목_막대)
            dpg.create_viewport(
                title=self._창_제목(), width=min(창_폭, 폭), height=min(창_높이, 높이 - 제목_막대),
                x_pos=x, y_pos=y + 제목_막대, vsync=False,
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
            self._배치를_맞춘다(dpg)

            while dpg.is_dearpygui_running():
                if self._한_바퀴(dpg):
                    dpg.render_dearpygui_frame()
                else:
                    # 할 일이 없으면 쉰다. 헛돌면 코어 하나를 다 먹어 온보드
                    # SW 의 모델이 쓸 몫을 뺏는다.
                    time.sleep(0.001)
        finally:
            self._입력을_적는다(dpg)
            stop_asr_server(self._전사_서버)
            stop_asr_server(self._감정_서버)
            dpg.destroy_context()
            for 것 in (self._시뮬, self._실물, self.launcher, self._거울):
                if 것 is not None:
                    것.close()

    def _입력을_되살린다(self, dpg, 값들: dict) -> None:
        """지난 입력값을 위젯에 넣는다. 고를 목록에서 빠진 값은 버리고 기본값을 둔다."""
        for 태그 in 기억할_입력:
            # 시뮬 정책은 시뮬을 만들 때 이미 지난 값으로 골랐다 (`main`).
            if 태그 not in 값들 or not dpg.does_item_exist(태그) or (태그 == "policy" and not self.real):
                continue
            목록 = dpg.get_item_configuration(태그).get("items")
            if 목록 and 값들[태그] not in 목록:
                continue
            dpg.set_value(태그, 값들[태그])
        # 위젯 밖에 든 값을 맞춘다. 콜백은 부르지 않는다 — 젯슨에는 송출에 붙을 때 보낸다.
        self._경로 = dpg.get_value("link_route") or self._경로
        if self.launcher is not None:
            self.launcher.set_route(self._경로)
        self._볼륨 = int(dpg.get_value("real_volume"))
        self._수동_속도값 = {키: float(dpg.get_value(f"speed_{키}")) for 키 in self._수동_속도값}
        if self._영상들:
            이름들 = [self._영상_이름(영상) for 영상 in self._영상들]
            self._고른_영상 = 이름들.index(dpg.get_value("video_clip"))
            dpg.set_value("video_text", self._영상들[self._고른_영상].text)
        # 정책은 쪽마다 따로 기억한다. 시뮬 정책은 시뮬을 만들 때 이미 지난 값으로 골랐다 (`main`).
        self._실물_정책 = str(값들.get("run_policy", ""))
        if not self.real:
            # 시뮬은 띄울 때 실행 칸 설정으로 처음부터 돈다. 실물은 실행을 눌러야 돈다.
            self._시뮬을_띄운다(dpg)

    def _입력을_적는다(self, dpg) -> None:
        값들 = {태그: dpg.get_value(태그) for 태그 in 기억할_입력 if dpg.does_item_exist(태그)}
        if self.real:
            self._실물_정책 = dpg.get_value("run_policy")
        값들["run_policy"] = self._실물_정책
        값들["policy"] = (self._시뮬_정책_글(self._시뮬.policy) if self._시뮬 is not None
                       else 지난_입력().get("policy", ""))
        try:
            입력_기록.parent.mkdir(parents=True, exist_ok=True)
            입력_기록.write_text(json.dumps(값들, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _고른_모드(dpg) -> str:
        return next(m for m, 이름 in 모드_이름.items() if 이름 == dpg.get_value("verify_mode"))

    @property
    def sim(self):
        """지금 보이는 쪽이다. 시뮬이면 `Sim`, 실물이면 `Real`."""
        return self._실물 if self.real else self._시뮬

    def _창_제목(self) -> str:
        return "AI Rookie — 실물" if self.real else "AI Rookie — 시뮬레이터"

    # --- 루프 ---

    def _한_바퀴(self, dpg) -> bool:
        """때가 된 주기를 밀고, 화면 박자가 되었으면 그린다.

        그렸으면 참이다. 창은 그 때만 한 장을 내보낸다.
        """
        if self._바꿀_쪽 is not None:
            self._소스를_적용한다(dpg, self._바꿀_쪽)
            self._바꿀_쪽 = None
        지금 = self._시계()
        if self._시작 is None:
            self._시작 = 지금

        with self._소리_자물쇠:
            만든_것, self._만든_소리 = self._만든_소리, []
        for 소리 in 만든_것:
            self._소리를_보낸다(소리, "음성 합성", dpg)

        밀_주기 = self.sim.due(지금 - self._시작)
        if 밀_주기:
            self.sim.cycle(밀_주기)
            self._주기_시간.append(self.sim.cycle_ms)

        if 지금 - self._마지막_화면 < 1.0 / 화면_fps:
            return False
        self._마지막_화면 = 지금

        self._사람을_키로_움직인다(dpg)
        self._화면을_갱신한다(dpg)
        self._배치를_다듬는다(dpg)
        self._토글을_얹는다(dpg)
        self._조작_표시를_맞춘다(dpg)
        return True

    # --- 위젯 ---

    def _한글_글꼴을_문다(self, dpg) -> str | None:
        """한글 글꼴을 찾아 물린다. 못 찾으면 그냥 넘어간다."""
        from pathlib import Path

        for 경로 in 한글_글꼴_후보:
            if not Path(경로).exists():
                continue
            # 글자 범위를 따로 일러 주지 아니한다. dearpygui 2.x 가 자동으로
            # 잡으며 `add_font_range_hint` 는 아무 일도 하지 않는 채 남아 있다.
            with dpg.font_registry():
                글꼴 = dpg.add_font(경로, 글꼴_크기)
            dpg.bind_font(글꼴)
            return 경로
        return None

    def _화면을_짓는다(self, dpg) -> None:
        self.font_path = self._한글_글꼴을_문다(dpg)
        self._그림_위인가 = lambda: dpg.is_item_hovered("spectator_img")
        키 = {
            "W": dpg.mvKey_W, "A": dpg.mvKey_A, "S": dpg.mvKey_S,
            "D": dpg.mvKey_D, "Q": dpg.mvKey_Q, "E": dpg.mvKey_E,
            # 수동 조작의 머리 · 귀 (자율 사양 8장). 조종기 Y · A · X · B · View · LB · RB 자리다.
            "↑": dpg.mvKey_Up, "↓": dpg.mvKey_Down, "←": dpg.mvKey_Left, "→": dpg.mvKey_Right,
            "R": dpg.mvKey_R, "Z": dpg.mvKey_Z, "C": dpg.mvKey_C,
        }
        self._키가_눌렸나 = lambda 이름: dpg.is_key_down(키[이름])
        너비, 높이 = self.sim.views.robot_size
        구_너비, 구_높이 = self.sim.views.spectator_size

        # **3채널(mvFormat_Float_rgb)을 쓰지 아니한다.** 640×480 에서 Metal 이
        # 행 정렬을 못 맞춰 죽는다(AGX bytes_per_row 단언). 작은 텍스처에서는
        # 살아남아 작은 재현으로는 안 보인다. 실측은 research/02 에 있다.
        with dpg.texture_registry():
            dpg.add_raw_texture(
                너비, 높이, self._로봇_버퍼,
                format=dpg.mvFormat_Float_rgba, tag="robot_tex",
            )
            dpg.add_raw_texture(
                구_너비, 구_높이, self._구경꾼_버퍼,
                format=dpg.mvFormat_Float_rgba, tag="spectator_tex",
            )

        with dpg.window(tag="root", no_title_bar=True, no_move=True, no_resize=True, no_scrollbar=True):
            with dpg.group(horizontal=True):
                self._좌측(dpg)
                self._중앙(dpg)
                self._우측(dpg)
            self._아래(dpg)

        dpg.set_primary_window("root", True)
        dpg.set_viewport_resize_callback(lambda *_: self._배치를_맞춘다(dpg))

    def _배치를_맞춘다(self, dpg) -> None:
        """창 크기에서 세 자리의 폭과 그림 크기를 낸다.

        **두 시점이 스크롤 없이 한 화면에 들어와야 한다.** 그림 크기를 박아 두면
        창보다 커져 가운데에 스크롤이 생긴다. 텍스처는 그대로 두고 그리는
        크기만 줄인다.
        """
        창_폭 = dpg.get_viewport_client_width()
        # 실물은 아래에 실행 로그 패널이 있다. 위 세 자리는 그만큼 낮다.
        창_높이 = dpg.get_viewport_client_height() - 2 * 창_안쪽
        if self.real:
            창_높이 -= 로그_높이 + 위젯_사이

        중앙_폭 = max(320, 창_폭 - 좌측_폭 - 우측_폭 - 40)
        dpg.configure_item("left_panel", width=좌측_폭, height=창_높이)
        dpg.configure_item("center_panel", width=중앙_폭, height=창_높이)
        dpg.configure_item("right_panel", width=우측_폭, height=창_높이)

        라벨 = max(라벨_세로, dpg.get_item_rect_size("robot_label")[1])
        # 첫 계산은 어림이다. 패널 안쪽 여백과 위젯 사이 간격은 dearpygui 가
        # 정하므로 모델로 맞히려 하지 않는다. 그린 뒤 실제 자리를 재서
        # `_배치를_다듬는다` 가 고친다.
        수 = self._그림_수()
        self._그림_높이 = max(120, (창_높이 - 수 * 라벨 - (수 + 1) * 간격_목표) // 수)
        self._그림에_맞춘다(dpg, 중앙_폭)

    def _사람을_키로_움직인다(self, dpg) -> None:
        """WASD 로 걷게 하고 QE 로 돌린다. 수동 조종 중에는 사람 대신 로봇이다.
        실물에는 사람이 없다. 수동일 때 로봇 키만 젯슨으로 보낸다 (사양 4-4).

        **사람이 바라보는 쪽을 앞으로 삼는다.** 월드 축으로 움직이면 사람을
        돌려 놓은 뒤 W 가 엉뚱한 쪽으로 간다.
        """
        if self.real and not self.sim.manual:
            return
        지금 = self._시계()
        앞 = self._키_앞_시각
        self._키_앞_시각 = 지금

        앞뒤 = self._키가_눌렸나("W") - self._키가_눌렸나("S")
        좌우 = self._키가_눌렸나("D") - self._키가_눌렸나("A")
        돌기 = self._키가_눌렸나("Q") - self._키가_눌렸나("E")
        if self.sim.manual:
            # 로봇 조종기는 왼쪽이 + 다. S 는 후진이다. 화살표는 머리, R 은 머리 홈, Z · C 는 귀다.
            눌림 = self._키가_눌렸나
            self.sim.drive(
                forward=앞뒤, lateral=-좌우, turn=돌기,
                head_pitch=눌림("↑") - 눌림("↓"), head_yaw=눌림("←") - 눌림("→"),
                head_home=bool(눌림("R")), ear=(bool(눌림("Z")), bool(눌림("C"))),
            )
            return
        if 앞 is None:
            return
        dt = min(지금 - 앞, 0.1)  # 창이 멈췄다 돌아올 때 한 번에 튀지 않게 한다
        if not (앞뒤 or 좌우 or 돌기):
            return

        월드 = self.sim.world
        if 돌기:
            월드.set_person_yaw(월드.person_yaw + math.radians(사람_회전_dps) * 돌기 * dt)

        if 앞뒤 or 좌우:
            각 = 월드.person_yaw
            x, y = 월드.person_pos
            걸음 = 사람_속도_mps * dt
            # 앞이 (cos, sin) 이면 오른쪽은 (sin, -cos) 이다. 부호를 뒤집으면
            # D 가 왼쪽으로 간다.
            x += (math.cos(각) * 앞뒤 + math.sin(각) * 좌우) * 걸음
            y += (math.sin(각) * 앞뒤 - math.cos(각) * 좌우) * 걸음
            월드.set_person_pos(x, y)

        self._사람_손잡이를_맞춘다(dpg)

    def _사람_손잡이를_맞춘다(self, dpg) -> None:
        """키로 움직인 결과를 슬라이더에 되비춘다."""
        x, y = self.sim.world.person_pos
        dpg.set_value("person_x", x)
        dpg.set_value("person_y", y)
        dpg.set_value("person_yaw", math.degrees(self.sim.world.person_yaw))

    def _그림에_맞춘다(self, dpg, 중앙_폭: int) -> None:
        """정한 세로에 맞춰 그림과 간격 위젯을 고친다.

        두 그림은 세로가 같고 폭은 각자 비율을 따른다 — 로봇 시점 4:3, 구경꾼 2:1.
        넓은 쪽이 중앙 폭에 들어가도록 세로를 정한다. 실물은 로봇 시점 하나다.
        """
        비율 = {"robot_img": self.sim.views.robot_size[0] / self.sim.views.robot_size[1]}
        if not self.real:
            비율["spectator_img"] = self.sim.views.spectator_size[0] / self.sim.views.spectator_size[1]
        쓸_폭 = 중앙_폭 - 24

        그림_높이 = max(120, int(self._그림_높이))
        넓은_비 = max(비율.values())
        if 그림_높이 * 넓은_비 > 쓸_폭:
            그림_높이 = int(쓸_폭 / 넓은_비)
            self._그림_높이 = 그림_높이

        for 태그, 비 in 비율.items():
            폭 = int(그림_높이 * 비)
            dpg.configure_item(태그, width=폭, height=그림_높이)
            dpg.configure_item(태그.replace("_img", "_left_pad"), width=max(0, (쓸_폭 - 폭) // 2))
        # 맨 위에는 위젯 사이 간격이 붙지 않는다. 그만큼 더해야 셋이 같아진다.
        dpg.configure_item("center_top_pad", height=간격_목표 + 위젯_사이)
        dpg.configure_item("center_mid_pad", height=간격_목표)

    def _배치를_다듬는다(self, dpg) -> None:
        """아래에 남거나 모자란 만큼을 그림 세로에서 뺀다.

        위·가운데 간격은 고정이고 아래가 나머지를 받는다. 셋이 같아지려면
        아래 간격이 `간격_목표` 와 같아야 하므로, 그 차이를 두 그림이 나눠
        가진다. 한 프레임에 다 못 맞으면 다음 프레임에 또 줄인다.
        """
        패널_바닥 = dpg.get_item_rect_size("center_panel")[1]
        if 패널_바닥 <= 0:
            # 아직 한 번도 안 그렸다. 잴 것이 없다.
            return
        아래 = "robot_img" if self.real else "spectator_img"
        바닥 = dpg.get_item_pos(아래)[1] + dpg.get_item_rect_size(아래)[1]
        차이 = (패널_바닥 - 바닥) - 간격_목표
        if abs(차이) <= 2:
            return
        self._그림_높이 += 차이 / self._그림_수()
        self._그림에_맞춘다(dpg, dpg.get_item_rect_size("center_panel")[0])

    def _그림_수(self) -> int:
        return 1 if self.real else 2

    def _좌측(self, dpg) -> None:
        """실행 · 조작 · 표현과 소리는 시뮬과 실물이 같은 칸을 쓴다. 젯슨 연결은 실물만, 장면은 시뮬만 보인다."""
        with dpg.child_window(tag="left_panel", width=좌측_폭, height=400):
            dpg.add_radio_button(
                ["시뮬", "실물"], tag="source", default_value="실물" if self.real else "시뮬",
                horizontal=True, callback=self._소스를_바꾼다,
            )
            with dpg.group(tag="real_group", show=self.real):
                self._실물_자리(dpg)
            self._묶음(dpg, "실행", self._실행_자리)
            self._묶음(dpg, "조작", self._조작)
            self._묶음(dpg, "표현 · 소리", self._표현_소리)
            with dpg.group(tag="sim_group", show=not self.real):
                self._묶음(dpg, "시뮬 장면", self._시뮬_자리)
            dpg.add_separator()
            dpg.add_text("", tag="debug_status", wrap=좌측_폭 - 20, color=상태_색)

    @staticmethod
    def _묶음(dpg, 제목: str, 짓는다) -> None:
        """좌측 묶음 하나다. 구분선으로 나눈다. `제목` 은 코드에서 묶음을 가리키는 이름이다."""
        dpg.add_separator()
        짓는다(dpg)

    @staticmethod
    def _소제목(dpg, 글: str, 처음: bool = False) -> int:
        """묶음 안 덩어리의 소제목이다."""
        return dpg.add_text(글, color=소제목_색)

    def _실물_자리(self, dpg) -> None:
        """젯슨 연결이다. 실물만 쓴다."""
        self._소제목(dpg, "젯슨 연결", 처음=True)
        dpg.add_text("", tag="link_status", wrap=좌측_폭 - 20, color=상태_색)
        dpg.add_radio_button(["같은 망", "NetBird"], tag="link_route", default_value=self._경로,
                             horizontal=True, callback=self._경로를_바꾼다)
        with dpg.tooltip("link_route"):
            dpg.add_text("젯슨에 붙는 길. 고른 쪽으로만 붙는다\n"
                         "같은 망: 맥과 같은 와이파이 · 핫스팟에서 젯슨 호스트 키가 맞는 주소를 찾는다\n"
                         "NetBird: ssh 별칭 jetson")

    def _실행_자리(self, dpg) -> None:
        """실물 진입점의 실행 인자다. 시뮬도 같은 칸을 실행할 때 건다 -- 실물은 젯슨에서, 시뮬은 이 앱에서 처음부터 돈다."""
        self._소제목(dpg, "걷기 정책", 처음=True)
        dpg.add_combo(
            [] if self.real else self._시뮬_정책_목록(), tag="run_policy", width=-1,
            default_value="" if self.real else self._시뮬_정책_글(self._시뮬.policy),
        )
        with dpg.tooltip("run_policy"):
            dpg.add_text("걷기 정책. 실물은 젯슨 assets/, 시뮬은 Playground policies/ 의 것이다")
        self._소제목(dpg, "전사")
        dpg.add_radio_button(["기본 전사", "맥 Qwen"], tag="asr", default_value="기본 전사", horizontal=True)
        with dpg.tooltip("asr"):
            dpg.add_text("기본 전사: autonomy.json 의 asr_backend 그대로\n"
                         "맥 Qwen: --asr mac. 닫힌 발화를 이 맥의 Qwen3-ASR 로 보낸다. 맥이 답하지 않으면 로컬 대비로 전사\n"
                         "맥에서 Runtime tools/asr_mac_server.py 를 먼저 띄운다. 주소는 자동이다")
        self._소제목(dpg, "감정")
        dpg.add_radio_button(["기본 감정", "맥 em7"], tag="emotion", default_value="기본 감정", horizontal=True)
        with dpg.tooltip("emotion"):
            dpg.add_text("기본 감정: autonomy.json 의 emotion.backend 그대로 (실물은 젯슨, 시뮬은 이 앱 안의 em7)\n"
                         "맥 em7: --emotion mac. 감정 요청을 이 맥의 em7 감정 서버로 보낸다. 맥이 답하지 않으면 로컬 em7 로 추론\n"
                         "감정 서버(Runtime tools/emotion_mac_server.py)는 앱이 켜질 때 함께 뜬다. 주소는 자동이다")
        self._소제목(dpg, "서보 · 안전")
        dpg.add_checkbox(label="서보 구동", tag="run_arm")
        dpg.add_checkbox(label="서보 없이", tag="run_no_servo")
        with dpg.tooltip("run_arm"):
            dpg.add_text("--arm. 끄면 서보에 쓰지 않는다. 시뮬은 액추에이터 힘이 0 이라 축 늘어진다")
        with dpg.tooltip("run_no_servo"):
            dpg.add_text("--no-servo · 개발용. 서보 없이 카메라 · 마이크 · 모델 확인\n"
                         "센서는 홈 자세 값이다. 시뮬은 로봇을 홈 자세에 묶어 둔다")
        dpg.add_checkbox(label="안전 정지", tag="run_safety")
        dpg.add_input_int(label="서보 속도 상한", tag="run_servo_speed", default_value=0,
                          min_value=0, min_clamped=True, step=50, width=칸_폭)
        with dpg.tooltip("run_safety"):
            dpg.add_text("--safety · 안전 계층. IMU 끊김 · 넘어짐에서 이동 0")
        with dpg.tooltip("run_servo_speed"):
            dpg.add_text("--servo-speed · 46번 레지스터 step/s. 0 이면 상한 없음. 300 ≈ 26 °/s\n"
                         "실행할 때 정해 끝까지 이 값 하나다. 시뮬은 서보 목표가 한 주기에 움직이는 양을 자른다")
        dpg.add_button(label="실행", width=-1, callback=self._실행한다)
        dpg.add_button(label="정지", tag="stop_button", width=-1, callback=self._정지한다)
        with dpg.tooltip("stop_button"):
            dpg.add_text("Ctrl-C 와 같음. 홈 자세로 이동 후 토크 유지")
        dpg.add_text("", tag="run_state", wrap=좌측_폭 - 20, color=상태_색)

    def _시뮬_정책_목록(self) -> list[str]:
        return [self._시뮬_정책_글(p) for p in 정책들()]

    @staticmethod
    def _시뮬_정책_글(정책: Path) -> str:
        """정책 칸의 글이다. 목록 칸과 같은 꼴(외장 폴더부터)이어야 지난 입력으로 다시 고를 수 있다."""
        return str(정책.relative_to(정책_폴더)) if 정책.is_relative_to(정책_폴더) else 정책.name

    def _수동_속도(self, dpg) -> None:
        """수동 조작의 최대 속도다 (사양 4-4). 키를 누르면 이 속도로 걷는다. 바꾸면 바로 보낸다 -- 시뮬은 온보드 SW 에, 실물은 젯슨에."""
        수동 = load_autonomy()["manual"]
        with dpg.tooltip(self._소제목(dpg, "수동 속도", 처음=True)):
            dpg.add_text("앞 · 뒤 · 옆 m/s, 회전 rad/s. 상한은 학습 명령 범위. 앞뒤는 최소 속도 아래로 내지 않는다")
        for 키, 이름, 최소 in (("vx_max", "앞 m/s", 수동["vx_min"]), ("vx_back_max", "뒤 m/s", 수동["vx_min"]),
                             ("vy_max", "옆 m/s", 0.0), ("wz_max", "회전 rad/s", 0.0)):
            dpg.add_slider_float(
                label=이름, tag=f"speed_{키}", default_value=수동[키], min_value=최소,
                max_value=수동["speed_limit"][키], format="%.3f", width=칸_폭,
                user_data=키, callback=self._속도를_바꾼다,
            )

    def _속도를_바꾼다(self, sender=None, app_data=None, user_data=None) -> None:
        self._수동_속도값[user_data] = float(app_data)
        self._디버깅_알림(self._속도를_보낸다(),
                       "수동 속도 · " + " ".join(f"{v:.3f}" for v in self._수동_속도값.values()))

    def _속도를_보낸다(self) -> bool:
        return self.sim.send_command({"kind": "manual_speed", **self._수동_속도값})

    def _조작(self, dpg) -> None:
        """수동 속도와 판단 강제 이동이다 (사양 4-4 · 4-5). 시뮬은 온보드 SW 에, 실물은 젯슨 송출로 같은 명령을 보낸다."""
        self._수동_속도(dpg)
        self._소제목(dpg, "판단 강제 이동")
        dpg.add_button(label="APPROACH", tag="force_approach", width=-1, callback=self._접근으로_보낸다)
        dpg.add_button(label="IDLE", tag="force_idle", width=-1, callback=self._대기로_보낸다)
        with dpg.tooltip("force_approach"):
            dpg.add_text("종합 판단을 APPROACH 로 바로 보낸다")
        with dpg.tooltip("force_idle"):
            dpg.add_text("종합 판단을 IDLE 로 바로 보낸다")

    def _표현_소리(self, dpg) -> None:
        """표현 재생 · 효과음 · 소리 주입이다. 소리는 실물이면 젯슨 마이크에 섞고, 시뮬이면 시뮬 마이크로 넣는다."""
        self._소제목(dpg, "표현 재생", 처음=True)
        for 이름 in PRESET_NAMES:
            dpg.add_button(label=표현_이름[이름], width=-1, user_data=이름, callback=self._표현을_보낸다)

        self._소제목(dpg, "효과음")
        효과음들 = sorted(p.stem for p in 효과음_폴더.glob("*.wav"))
        dpg.add_combo(효과음들, tag="real_sound", default_value=효과음들[0] if 효과음들 else "", width=-1)
        dpg.add_button(label="재생", width=-1, callback=self._효과음을_낸다)
        dpg.add_slider_int(label="볼륨 %", tag="real_volume", default_value=100, min_value=0, max_value=100,
                           width=칸_폭, callback=self._볼륨을_바꾼다)
        with dpg.tooltip("real_volume"):
            dpg.add_text("모든 효과음에 곱한다. 음량 상한을 거친 소리 기준이다. 0 이면 내지 않는다")

        with dpg.tooltip(self._소제목(dpg, "소리 주입 · 감정 영상 소리")):
            dpg.add_text("실물은 젯슨 마이크에 섞고, 시뮬은 시뮬 마이크로 넣는다")
        self._주입_소리 = [(self._영상_이름(영상), 영상.wav) for 영상 in self._영상들]
        if self._주입_소리:
            dpg.add_combo([이름 for 이름, _ in self._주입_소리], tag="real_mic_source",
                          default_value=self._주입_소리[0][0], width=-1)
            dpg.add_button(label="넣기", width=-1, callback=self._소리를_넣는다)
        with dpg.tooltip(self._소제목(dpg, "소리 주입 · 음성 합성")):
            dpg.add_text("적은 문장을 맥 내장 음성으로 읽어 넣는다. 실물은 젯슨 마이크, 시뮬은 시뮬 마이크다")
        음성들 = sources.tts_voices()
        dpg.add_combo(list(음성들), tag="tts_voice", default_value=음성들[0], width=-1)
        dpg.add_input_text(tag="tts_text", hint="말할 문장", width=-1, on_enter=True,
                           callback=self._TTS를_읽는다)
        dpg.add_button(label="말하기", width=-1, callback=self._TTS를_읽는다)

    def _시뮬_자리(self, dpg) -> None:
        # 실물로 켜면 시뮬은 아직 없다. 위젯은 기본값으로 짓고 시뮬로 올 때 맞춘다.
        시뮬 = self._시뮬
        self._소제목(dpg, "사람 위치", 처음=True)
        사람_x, 사람_y = 시뮬.world.person_pos if 시뮬 else (0.0, 0.0)
        dpg.add_slider_float(
            label="x (m)", tag="person_x", default_value=사람_x,
            min_value=-5.0, max_value=5.0, callback=self._사람을_옮긴다, width=칸_폭,
        )
        dpg.add_slider_float(
            label="y (m)", tag="person_y", default_value=사람_y,
            min_value=-5.0, max_value=5.0, callback=self._사람을_옮긴다, width=칸_폭,
        )
        dpg.add_slider_float(
            label="방향 (°)", tag="person_yaw",
            default_value=math.degrees(시뮬.world.person_yaw) if 시뮬 else 0.0,
            min_value=-180.0, max_value=180.0, callback=self._사람을_돌린다, width=칸_폭,
        )
        dpg.add_button(label="로봇 쪽으로 돌리기", width=-1, callback=self._사람이_로봇을_보게)
        self._소제목(dpg, "서보 전압 · 온도 주입")
        dpg.add_slider_float(
            label="전압 (V)", tag="inject_v", default_value=Bridge.기본_전압_V,
            min_value=5.0, max_value=8.0, callback=self._전압을_넣는다, width=칸_폭,
        )
        dpg.add_slider_float(
            label="온도 (°C)", tag="inject_c", default_value=Bridge.기본_온도_C,
            min_value=20.0, max_value=90.0, callback=self._온도를_넣는다, width=칸_폭,
        )
        dpg.add_button(label="기본값", width=-1, callback=self._주입을_지운다)

        self._모드와_소스(dpg)

    def _모드와_소스(self, dpg) -> None:
        """검증 모드와 발화·소리·영상 소스를 고르는 자리다 (사양 3-3)."""
        self._소제목(dpg, "카메라 입력")
        dpg.add_radio_button(
            [모드_이름[m] for m in 모드들], tag="verify_mode", default_value=모드_이름[모드_추종],
            horizontal=True, callback=self._모드를_바꾼다,
        )
        self._소제목(dpg, "감정 영상 (얼굴)")
        if self._영상들:
            dpg.add_combo(
                [self._영상_이름(영상) for 영상 in self._영상들],
                tag="video_clip", default_value=self._영상_이름(self._영상들[0]),
                width=-1, callback=self._영상을_고른다,
            )
            with dpg.tooltip("video_clip"):
                dpg.add_text(self._영상들[0].text, tag="video_text", wrap=300)
            dpg.add_button(
                label="영상만 재생", width=-1, callback=self._영상_그림만_튼다
            )
            dpg.add_button(
                label="영상·소리 재생", width=-1, callback=self._영상_소리와_그림_튼다
            )
        else:
            dpg.add_text("영상 없음", wrap=좌측_폭 - 20)
        self._소제목(dpg, "맥북 장치")
        # **기본 장치를 그냥 쓰지 아니한다.** 이어폰이 붙어 있으면 macOS 의
        # 기본 입력이 그 쪽이라 "맥북 마이크" 버튼이 맥북 마이크를 안 연다.
        self._마이크_장치 = sources.input_devices()
        if self._마이크_장치:
            이름들 = [이름 for _, 이름 in self._마이크_장치]
            고를_것 = next(
                (이름 for 이름 in 이름들 if "MacBook" in 이름), 이름들[0]
            )
            dpg.add_combo(
                이름들, tag="mic_device", default_value=고를_것, width=-1
            )
        dpg.add_button(label="마이크 켜기", width=-1, callback=self._마이크를_켠다)
        # OpenCV 는 카메라 이름을 주지 아니한다. 번호로 고른다 -- AVFoundation
        # 차례이며 0 이 맥북 내장이다.
        dpg.add_input_int(
            label="카메라 번호", tag="cam_index", default_value=0,
            min_value=0, max_value=8, min_clamped=True, max_clamped=True, width=칸_폭,
        )
        dpg.add_button(label="카메라 켜기", width=-1, callback=self._카메라를_켠다)
        dpg.add_button(label="입력 멈춤", width=-1, callback=self._주입을_멈춘다)
        dpg.add_text("", tag="inject_status", wrap=좌측_폭 - 20, color=상태_색)

    def _중앙(self, dpg) -> None:
        with dpg.child_window(tag="center_panel", width=700, height=400):
            dpg.add_spacer(height=0, tag="center_top_pad")
            with dpg.group(horizontal=True):
                dpg.add_text("로봇 카메라", tag="robot_label")
                # 글자와 높이가 같은 작은 단추다. 배치가 라벨 높이로 잰다.
                dpg.add_button(label="창 복제", tag="mirror_button", small=True, callback=self._거울을_바꾼다)
            with dpg.tooltip("mirror_button"):
                dpg.add_text("로봇 카메라 화면만 딴 창에 띄운다. 다시 누르면 닫는다. 그 창에서 F 는 전체 화면")
            # 두 그림은 중앙 자리 가운데에 둔다. 왼쪽 여백은 `_그림에_맞춘다` 가 정한다.
            with dpg.group(horizontal=True, horizontal_spacing=0):
                dpg.add_spacer(width=0, tag="robot_left_pad")
                dpg.add_image("robot_tex", tag="robot_img")
            # 실물에는 외부 시점을 두지 않는다. 로봇 카메라가 중앙 세로를 다 쓴다.
            with dpg.group(tag="spectator_group", show=not self.real):
                dpg.add_spacer(height=0, tag="center_mid_pad")
                dpg.add_text("외부 시점", tag="spectator_label")
                with dpg.group(horizontal=True, horizontal_spacing=0):
                    dpg.add_spacer(width=0, tag="spectator_left_pad")
                    dpg.add_image("spectator_tex", tag="spectator_img")
            # 구경꾼 시점 우측 상단에 얹는다. 실물은 로봇 시점이다. 자리는 그림을 그린 뒤 맞춘다.
            dpg.add_button(
                label=self._조작_이름(), tag="manual_toggle", width=토글_폭,
                callback=self._조작을_바꾼다,
            )
            # 키 안내는 화면에 늘어놓지 않고 여기에만 둔다.
            with dpg.tooltip("manual_toggle", tag="manual_tip"):
                dpg.add_text(self._키_안내(), tag="key_hint")

        # 창 전체에서 오는 이벤트다. 그림 위인지 여기서 가린다.
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(button=0, callback=self._구경꾼을_잡는다)
            dpg.add_mouse_release_handler(button=0, callback=self._구경꾼을_놓는다)
            dpg.add_mouse_drag_handler(button=0, callback=self._구경꾼을_돌린다)
            dpg.add_mouse_wheel_handler(callback=self._구경꾼을_당긴다)

    def _아래(self, dpg) -> None:
        """실물의 실행 로그다. 창 폭 전체를 쓰는 아래 패널이다 (사양 4-4)."""
        with dpg.child_window(tag="bottom_panel", height=-1, show=self.real):
            dpg.add_text("실행 로그")
            with dpg.child_window(tag="run_log_box", height=-1, border=False):
                dpg.add_text("", tag="run_log", wrap=0)

    def _우측(self, dpg) -> None:
        """맨 위 요약 · 최근 기록 · 계층 상세 · 센서 순이다. 구분선으로 나눈다."""
        with dpg.child_window(tag="right_panel", width=우측_폭, height=400):
            dpg.add_text("", tag="clock")
            dpg.add_separator()
            with dpg.table(header_row=False, borders_innerV=True, policy=dpg.mvTable_SizingFixedFit):
                dpg.add_table_column()
                dpg.add_table_column()
                for i, 이름 in enumerate(요약_칸):
                    with dpg.table_row():
                        dpg.add_text(이름)
                        dpg.add_text("—", tag=f"summary_{i}")
            dpg.add_separator()
            dpg.add_text("최근 기록")
            with dpg.child_window(tag="history_box", height=140):
                dpg.add_text("", tag="history", wrap=400)
            dpg.add_separator()
            dpg.add_group(tag="status")

    # --- 갱신 ---

    def _화면을_갱신한다(self, dpg) -> None:
        """그림은 매 프레임, 글자는 늦춰서 갱신한다."""
        self._거울을_챙긴다()
        if self.sim.frame is not None:
            self._텍스처에_넣는다(self._로봇_버퍼, self.sim.frame)
            # 화면에만 그린다. 온보드 SW 가 받은 프레임(`sim.frame`)은 건드리지
            # 아니한다 -- 중앙 위는 들어간 것과 같은 프레임이어야 한다 (사양 4-1).
            너비, 높이 = self.sim.views.robot_size
            조회 = self.sim.runtime.snapshot()
            self._박스를_그린다(
                self._로봇_버퍼, 너비, 높이, box_rects(조회, 너비, 높이), target_id(조회)
            )
            dpg.set_value("robot_tex", self._로봇_버퍼)
            self._거울에_넣는다()
        if not self.real:
            self._텍스처에_넣는다(
                self._구경꾼_버퍼,
                self.sim.views.spectator_frame(
                    self.sim.world.data, self.sim.bridge.ear, self.sim.bridge.eye
                ),
            )
            dpg.set_value("spectator_tex", self._구경꾼_버퍼)

        지금 = self._시계()
        if 지금 - self._마지막_글자 < 글자_갱신_주기_s:
            return
        self._마지막_글자 = 지금

        dpg.set_value("clock", self._시계_글자())
        if self.real:
            self._실물_글자를_그린다(dpg)
        else:
            self._시뮬_글자를_그린다(dpg)
        self._상태를_그린다(dpg)
        self._주기_시간.clear()

    def _거울을_바꾼다(self, sender=None, app_data=None, user_data=None) -> None:
        """누른 것만 적는다. 콜백은 딴 실이라 여기서 닫으면 고리가 쓰던 공유 메모리가 사라진다."""
        self._거울_바꿈 = True

    def _거울을_챙긴다(self) -> None:
        """단추로 열고 닫는다. 그 창을 사람이 닫았으면 치운다."""
        켜짐 = self._거울 is not None and self._거울.alive
        if self._거울 is not None and (self._거울_바꿈 or not 켜짐):
            self._거울.close()
            self._거울 = None
        if self._거울_바꿈 and not 켜짐:
            self._거울 = Mirror(*self.sim.views.robot_size)
            self._거울.put(self._로봇_버퍼)
        self._거울_바꿈 = False

    def _거울에_넣는다(self) -> None:
        if self._거울 is not None:
            self._거울.put(self._로봇_버퍼)

    def _시계_글자(self) -> str:
        한도 = 제어_주기_s * 1000.0
        if self._주기_시간:
            평균 = sum(self._주기_시간) / len(self._주기_시간)
            최대 = max(self._주기_시간)
            주기 = f"주기 소요 평균 {평균:5.2f} · 최대 {최대:5.2f} ms (한도 {한도:.0f} ms)"
        else:
            주기 = f"주기 소요 -- (한도 {한도:.0f} ms)"
        return (
            f"시각 {self.sim.t:7.2f} s   주기 {self.sim.cycles}   "
            f"늦은 주기 {self.sim.missed}\n{주기}"
        )

    def _상태를_그린다(self, dpg) -> None:
        """조회를 표 위젯으로 그린다.

        **공백으로 칸을 맞추지 아니한다.** 글꼴이 고정폭이 아니고, 고정폭이라도
        한글은 너비가 두 배라 어긋난다. 칸 맞추기는 표 위젯이 할 일이다.

        **갱신할 때 표를 다시 만들지 아니한다.** 새로 만든 표는 내용을 아직
        재지 못해 그 프레임에 열이 최소 폭으로 찌그러지고 다음 프레임에 제대로
        잡힌다. 0.3 초마다 다시 만들면 그 한 프레임이 깜박임으로 보인다.
        모양이 바뀔 때만 다시 만들고, 평소에는 칸 값만 갈아 끼운다.
        """
        덩어리들 = self._차례(self.sim.status())
        요약 = dict(self.sim.summary())
        for i, 이름 in enumerate(요약_칸):
            dpg.set_value(f"summary_{i}", 요약.get(이름, "—"))
        dpg.set_value("history", "\n".join(self.sim.history.lines))
        모양 = tuple(
            (덩어리.title, 덩어리.header, len(덩어리.rows), self._열수(덩어리))
            for 덩어리 in 덩어리들
        )
        if 모양 != self._표_모양:
            self._표를_짓는다(dpg, 덩어리들)
            self._표_모양 = 모양

        for si, 덩어리 in enumerate(덩어리들):
            for ri, 행 in enumerate(덩어리.rows):
                for ci, 칸 in enumerate(행):
                    dpg.set_value(f"cell_{si}_{ri}_{ci}", 칸)

    @staticmethod
    def _차례(덩어리들) -> tuple:
        """계층 상세를 앞에, 서보 표 · IMU 를 뒤에 둔다. 운영 중에는 계층을 더 자주 본다."""
        센서 = {"서보", "IMU"}
        return tuple(d for d in 덩어리들 if d.title not in 센서) + tuple(d for d in 덩어리들 if d.title in 센서)

    @staticmethod
    def _열수(덩어리) -> int:
        if 덩어리.header is not None:
            return len(덩어리.header)
        return max(len(행) for 행 in 덩어리.rows)

    def _표를_짓는다(self, dpg, 덩어리들) -> None:
        """표를 처음 만들거나, 계층이 서서 줄 수가 바뀌었을 때 다시 만든다."""
        dpg.delete_item("status", children_only=True)

        if not 덩어리들:
            dpg.add_text("조회 없음", parent="status")
            return

        for si, 덩어리 in enumerate(덩어리들):
            dpg.add_text(덩어리.title, parent="status")
            열수 = self._열수(덩어리)
            with dpg.table(
                parent="status",
                header_row=덩어리.header is not None,
                borders_innerV=True,
                borders_outerV=True,
                borders_outerH=True,
                policy=dpg.mvTable_SizingFixedFit,
            ):
                for ci in range(열수):
                    dpg.add_table_column(
                        label=덩어리.header[ci] if 덩어리.header is not None else ""
                    )
                for ri, 행 in enumerate(덩어리.rows):
                    with dpg.table_row():
                        for ci, 칸 in enumerate(행):
                            dpg.add_text(칸, tag=f"cell_{si}_{ri}_{ci}")
            dpg.add_spacer(height=6, parent="status")

    @staticmethod
    def _빈_텍스처(너비: int, 높이: int) -> np.ndarray:
        """RGBA 평면 버퍼다. 알파는 1 로 채워 두고 다시 건드리지 아니한다."""
        버퍼 = np.zeros(높이 * 너비 * 4, dtype=np.float32)
        버퍼[3::4] = 1.0
        return 버퍼

    @staticmethod
    def _박스를_그린다(
        버퍼: np.ndarray, 너비: int, 높이: int, 박스들, 고른_id=None, 두께: int = 3
    ) -> None:
        """텍스처 버퍼에 박스 테두리를 긋는다. 검출은 초록, 종합 판단이 고른 대상은 노랑이다."""
        그림 = 버퍼.reshape(높이, 너비, 4)
        for 번호, x1, y1, x2, y2 in 박스들:
            색 = (1.0, 0.85, 0.0) if 번호 == 고른_id else (0.2, 1.0, 0.2)
            x1, x2 = max(0, x1), min(너비, x2)
            y1, y2 = max(0, y1), min(높이, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            for 칸, 값 in enumerate(색):
                그림[y1 : y1 + 두께, x1:x2, 칸] = 값
                그림[max(y1, y2 - 두께) : y2, x1:x2, 칸] = 값
                그림[y1:y2, x1 : x1 + 두께, 칸] = 값
                그림[y1:y2, max(x1, x2 - 두께) : x2, 칸] = 값

    @staticmethod
    def _텍스처에_넣는다(버퍼: np.ndarray, frame: np.ndarray) -> None:
        """uint8 RGB 프레임을 텍스처 버퍼의 색 자리에 덮어쓴다."""
        높이, 너비 = frame.shape[:2]
        버퍼.reshape(높이, 너비, 4)[:, :, :3] = frame.astype(np.float32) / 255.0

    # --- 위젯이 부르는 것 ---

    def _초기화된_화면(self) -> None:
        """초기화 뒤에 벽시계 기준과 위젯을 되돌린다."""
        import dearpygui.dearpygui as dpg

        self._시작 = None
        self._주기_시간.clear()
        self._사람_손잡이를_맞춘다(dpg)
        dpg.set_value("inject_v", Bridge.기본_전압_V)
        dpg.set_value("inject_c", Bridge.기본_온도_C)

    def _사람을_옮긴다(self, sender, app_data) -> None:
        import dearpygui.dearpygui as dpg

        self.sim.world.set_person_pos(dpg.get_value("person_x"), dpg.get_value("person_y"))

    def _사람을_돌린다(self, sender, app_data) -> None:
        self.sim.world.set_person_yaw(math.radians(float(app_data)))

    def _사람이_로봇을_보게(self, sender=None, app_data=None, user_data=None) -> None:
        import dearpygui.dearpygui as dpg

        각 = self.sim.world.로봇을_바라보는_각()
        self.sim.world.set_person_yaw(각)
        dpg.set_value("person_yaw", math.degrees(각))

    def _전압을_넣는다(self, sender, app_data) -> None:
        self.sim.injector.drop_voltage(float(app_data))

    def _온도를_넣는다(self, sender, app_data) -> None:
        self.sim.injector.raise_temperature(float(app_data))

    def _주입을_지운다(self, sender=None, app_data=None, user_data=None) -> None:
        import dearpygui.dearpygui as dpg

        self.sim.injector.clear()
        dpg.set_value("inject_v", Bridge.기본_전압_V)
        dpg.set_value("inject_c", Bridge.기본_온도_C)

    # --- 검증 모드와 입력원 ---

    def _알린다(self, 글: str) -> None:
        import dearpygui.dearpygui as dpg

        dpg.set_value("inject_status", 글)

    def _키_안내(self) -> str:
        if not (self.real or self.sim.manual):
            return "사람: W·S 앞뒤   A·D 옆   Q·E 회전"
        return "로봇: W·S 앞뒤   A·D 옆   Q·E 회전   화살표 머리   R 머리 홈   Z·C 귀"

    def _조작_이름(self) -> str:
        return "수동 조작 끄기" if self.sim.manual else "수동 조작 켜기"

    def _조작을_바꾼다(self, sender=None, app_data=None, user_data=None) -> None:
        import dearpygui.dearpygui as dpg

        켬 = not self.sim.manual
        글 = "수동 조작 켬" if 켬 else "수동 조작 끔"
        if self.real:
            # 젯슨이 바꾸고 조회로 알려 주면 `_조작_표시를_맞춘다` 가 토글을 고친다.
            self._디버깅_알림(self.sim.set_manual(켬), 글)
            return
        self.sim.set_manual(켬)
        self._조작_표시를_맞춘다(dpg)
        self._알린다(글)

    def _조작_표시를_맞춘다(self, dpg) -> None:
        """토글 글자와 키 안내를 지금 쪽에 맞춘다. 실물은 조종기 Menu 로도 바뀐다. 바뀔 때만 고친다."""
        표시 = (self._조작_이름(), self._키_안내())
        if 표시 != self._조작_표시:
            dpg.configure_item("manual_toggle", label=표시[0])
            dpg.set_value("key_hint", 표시[1])
            self._조작_표시 = 표시

    # --- 시뮬 · 실물 (사양 4-4) ---

    def _소스를_바꾼다(self, sender=None, app_data=None, user_data=None) -> None:
        """고른 쪽을 적어 두기만 한다.

        **콜백에서 바꾸지 아니한다.** dearpygui 는 콜백을 딴 실에서 부르는데, 바꾸는 일은 MuJoCo
        렌더러를 만들고 macOS 는 그것을 메인 실에서만 허락한다. 콜백에서 만들면 창이 멈춘 채 죽는다.
        """
        self._바꿀_쪽 = app_data

    def _고른_전사(self, dpg, 실물: bool) -> tuple[str, str]:
        """(전사 백엔드, 맥 주소). 기본 전사면 둘 다 비운다.

        전사 서버는 이 앱과 같은 맥에서 돈다. 그래서 주소는 정해져 있다 -- 실물은 젯슨에 붙은 경로대로
        이 맥의 NetBird 주소나 와이파이 주소, 시뮬은 127.0.0.1 이다.
        """
        if dpg.get_value("asr") != "맥 Qwen":
            return "", ""
        return "mac", asr_url(route=self._경로) if 실물 else f"http://127.0.0.1:{맥_전사_포트}"

    def _고른_감정(self, dpg, 실물: bool) -> tuple[str, str]:
        """(감정 추론 자리, 맥 주소). 기본 감정이면 둘 다 비운다. 주소 규칙은 `_고른_전사` 와 같다."""
        if dpg.get_value("emotion") != "맥 em7":
            return "", ""
        return "mac", asr_url(route=self._경로, port=맥_감정_포트) if 실물 else f"http://127.0.0.1:{맥_감정_포트}"

    def _경로를_바꾼다(self, sender=None, app_data=None, user_data=None) -> None:
        """젯슨에 붙는 경로를 바꾼다. 실행기는 다음 바퀴에 새 경로로 다시 찾는다."""
        self._경로 = app_data
        if self.launcher is not None:
            self.launcher.set_route(app_data)

    def _소스를_적용한다(self, dpg, 쪽: str) -> None:
        """시뮬과 실물을 오간다. 정책 칸은 쪽마다 고른 것을 따로 든다. 시뮬로 돌아오면 실행 칸 설정으로 처음부터 돈다."""
        실물로 = 쪽 == "실물"
        if 실물로 == self.real:
            return
        if 실물로:
            self._시뮬_정책 = dpg.get_value("run_policy")
            self._실물 = Real(host=None)
            self.launcher = Launcher(self.host, tunnel_port=self._실물.port, route=self._경로)
            dpg.configure_item("run_policy", items=[])
            dpg.set_value("run_policy", self._실물_정책)
            self.real = True
        else:
            self._실물_정책 = dpg.get_value("run_policy")
            self._실물.close()
            self.launcher.close()
            self._실물 = self.launcher = None
            dpg.configure_item("run_policy", items=self._시뮬_정책_목록())
            dpg.set_value("run_policy", self._시뮬_정책 or self._시뮬_정책_글(
                self._시뮬.policy if self._시뮬 is not None else 기본_정책))
            self.real = False
            self._시뮬을_띄운다(dpg)
        self._모드에_맞춘다(dpg)

    def _모드에_맞춘다(self, dpg) -> None:
        """보이는 쪽에 맞춰 자리를 바꾸고 시계 · 표를 새로 잡는다."""
        dpg.configure_item("real_group", show=self.real)
        dpg.configure_item("sim_group", show=not self.real)
        dpg.configure_item("spectator_group", show=not self.real)
        dpg.configure_item("bottom_panel", show=self.real)
        self._배치를_맞춘다(dpg)
        dpg.set_viewport_title(self._창_제목())
        self._표_모양 = None
        self._토글_자리 = None
        # 앞 쪽의 마지막 그림이 남지 않게 로봇 시점을 비운다. 새 프레임이 오면 덮인다.
        self._로봇_버퍼[:] = self._빈_텍스처(*self.sim.views.robot_size)
        dpg.set_value("robot_tex", self._로봇_버퍼)
        if self.real:
            self._시작 = None
            self._주기_시간.clear()
        else:
            dpg.set_value("policy", str(self.sim.policy.relative_to(정책_폴더)))
            self._초기화된_화면()
            self._알린다("시뮬 전환")

    def _실행한다(self, sender=None, app_data=None, user_data=None) -> None:
        """실행 칸의 설정으로 띄운다. 실물은 젯슨에서, 시뮬은 이 앱에서 처음부터 돈다."""
        import dearpygui.dearpygui as dpg

        if not self.real:
            self._시뮬을_띄운다(dpg)
            return
        정책 = dpg.get_value("run_policy")
        if not 정책:
            self.launcher.note = "정책 선택 필요"
            return
        전사, 주소 = self._고른_전사(dpg, 실물=True)
        감정, 감정_주소 = self._고른_감정(dpg, 실물=True)
        if (전사 and not 주소) or (감정 and not 감정_주소):
            self.launcher.note = f"맥 주소 없음 · {self._경로} 연결 확인"
            return
        self.launcher.start(
            정책, arm=dpg.get_value("run_arm"), safety=dpg.get_value("run_safety"),
            no_servo=dpg.get_value("run_no_servo"), servo_speed=dpg.get_value("run_servo_speed"),
            asr=전사, asr_url=주소, emotion=감정, emotion_url=감정_주소,
        )

    def _시뮬을_띄운다(self, dpg) -> None:
        """실행 칸의 설정으로 시뮬을 처음부터 돌린다. 전사 · 감정이 바뀌었으면 지각 · 감정을 새로 띄운다(모델을 다시 올린다)."""
        고른 = 정책_폴더 / str(dpg.get_value("run_policy"))
        정책 = 고른 if 고른.is_file() else (self._시뮬.policy if self._시뮬 is not None else 기본_정책)
        전사 = self._고른_전사(dpg, 실물=False) + self._고른_감정(dpg, 실물=False)
        if self._시뮬 is not None and self._시뮬.asr_choice != 전사:
            self._시뮬.close()
            self._시뮬 = None
        if self._시뮬 is None:
            self._시뮬 = Sim(policy=정책, asr=전사[0], asr_url=전사[1], emotion=전사[2], emotion_url=전사[3])
        self._시뮬.set_mode(self._고른_모드(dpg))
        self._시뮬.launch(
            정책, arm=dpg.get_value("run_arm"), safety=dpg.get_value("run_safety"),
            no_servo=dpg.get_value("run_no_servo"), servo_speed=dpg.get_value("run_servo_speed"),
        )
        # 수동 속도 · 볼륨은 실행과 무관하게 바로 거는 값이다. 새로 지은 시뮬에도 건다.
        self._시뮬.send_command({"kind": "manual_speed", **self._수동_속도값})
        self._시뮬.send_command({"kind": "volume", "value": self._볼륨 / 100})
        self._초기화된_화면()
        self._알린다(f"실행 · {정책.name} · 전사 {self._시뮬.asr} · 감정 {self._시뮬.emotion}")

    def _정지한다(self, sender=None, app_data=None, user_data=None) -> None:
        """Ctrl-C 와 같다. 홈 자세로 가서 쥔다."""
        if self.real:
            self.launcher.stop()
        else:
            self._시뮬.stop()
            self._알린다("정지")

    def _시뮬_글자를_그린다(self, dpg) -> None:
        시뮬 = self._시뮬
        dpg.set_value("run_state", f"{'실행 중' if 시뮬.running else '정지'} · {시뮬.policy.name} · 전사 {시뮬.asr} · 감정 {시뮬.emotion}")

    def _실물_글자를_그린다(self, dpg) -> None:
        실행기 = self.launcher
        젯슨 = f"연결됨 · {실행기.address}" if 실행기.connected else 실행기.note
        정책 = "없음" if self.sim.policy.name == "-" else self.sim.policy.name
        dpg.set_value("link_status", f"젯슨: {젯슨}\n화면 전송: {self.sim.state}\n실행 정책: {정책}")
        목록 = 실행기.policies
        if dpg.get_item_configuration("run_policy")["items"] != 목록:
            dpg.configure_item("run_policy", items=목록)
        if 목록 and dpg.get_value("run_policy") not in 목록:
            # 이미 도는 실행이 있으면 그 정책을, 없으면 기본 정책을 고른 채로 둔다.
            # 지난번에 고른 정책이 젯슨에 없어졌어도 여기로 온다.
            if self.sim.policy.name in 목록:
                dpg.set_value("run_policy", self.sim.policy.name)
            elif self._실물_정책 in 목록:
                dpg.set_value("run_policy", self._실물_정책)
            elif 기본_정책_이름 in 목록:
                dpg.set_value("run_policy", 기본_정책_이름)
        dpg.set_value("run_state", f"{'실행 중' if 실행기.running else '정지'}   {실행기.note if 실행기.connected else ''}")
        dpg.set_value("run_log", "\n".join(실행기.lines))
        dpg.set_y_scroll("run_log_box", dpg.get_y_scroll_max("run_log_box"))
        # 젯슨이 다시 뜨면 볼륨 · 수동 속도가 기본값이다. 송출에 새로 붙을 때마다 고른 값을 보낸다.
        if self.sim.state == "받는 중" and self._앞_전송 != "받는 중":
            self._볼륨을_보낸다()
            self._속도를_보낸다()
        self._앞_전송 = self.sim.state

    # --- 실물 디버깅 도구 (사양 4-5) ---

    def _디버깅_알림(self, 보냄: bool, 글: str, dpg=None) -> None:
        if dpg is None:
            import dearpygui.dearpygui as dpg

        못함 = "못 보냄 — 송출에 붙어 있지 않다" if self.real else "못 따름 — 터미널 출력을 본다"
        dpg.set_value("debug_status", 글 if 보냄 else f"{못함} ({글})")

    def _소리를_넣는다(self, sender=None, app_data=None, user_data=None) -> None:
        import dearpygui.dearpygui as dpg

        이름 = dpg.get_value("real_mic_source")
        소리_목록 = dict(self._주입_소리)
        if 이름 not in 소리_목록:
            return
        try:
            소리 = sources.WavFile.from_path(소리_목록[이름])
        except Exception as 오류:
            dpg.set_value("debug_status", f"소리 열기 실패: {오류}")
            return
        self._소리를_보낸다(소리, f"소리 주입 · {이름}")

    def _소리를_보낸다(self, 소리, 글: str, dpg=None) -> None:
        """실물은 젯슨 마이크에 섞고, 시뮬은 시뮬 마이크로 넣는다. 고리에서 부르면 그 `dpg` 를 넘긴다."""
        if self.real:
            보냄 = self.sim.send_mic(sources.to_mic(소리.samples))
        else:
            self.sim.injector.play(t=self.sim.t, audio=소리)
            보냄 = True
        self._디버깅_알림(보냄, f"{글} {소리.duration_s:.1f} s", dpg)

    def _접근으로_보낸다(self, sender=None, app_data=None, user_data=None) -> None:
        self._디버깅_알림(self.sim.send_command({"kind": "approach"}), "APPROACH 강제 이동")

    def _대기로_보낸다(self, sender=None, app_data=None, user_data=None) -> None:
        self._디버깅_알림(self.sim.send_command({"kind": "idle"}), "IDLE 강제 이동")

    def _표현을_보낸다(self, sender=None, app_data=None, user_data=None) -> None:
        보냄 = self.sim.send_command({"kind": "expression", "name": user_data})
        self._디버깅_알림(보냄, f"표현 재생 · {표현_이름[user_data]}")

    def _볼륨을_바꾼다(self, sender=None, app_data=None, user_data=None) -> None:
        self._볼륨 = int(app_data)
        self._디버깅_알림(self._볼륨을_보낸다(), f"볼륨 · {self._볼륨} %")

    def _볼륨을_보낸다(self) -> bool:
        return self.sim.send_command({"kind": "volume", "value": self._볼륨 / 100})

    def _효과음을_낸다(self, sender=None, app_data=None, user_data=None) -> None:
        import dearpygui.dearpygui as dpg

        이름 = dpg.get_value("real_sound")
        if not 이름:
            return
        보냄 = self.sim.send_command({"kind": "sound", "name": 이름})
        self._디버깅_알림(보냄, f"효과음 · {이름}")

    def _토글을_얹는다(self, dpg) -> None:
        """조작 토글을 구경꾼 그림(실물은 로봇 시점)의 우측 상단에 둔다. 자리가 바뀔 때만 옮긴다."""
        그림 = "robot_img" if self.real else "spectator_img"
        x, y = dpg.get_item_pos(그림)
        폭, _ = dpg.get_item_rect_size(그림)
        자리 = [int(x + 폭 - 토글_폭 - 토글_여백), int(y + 토글_여백)]
        if 자리 != self._토글_자리:
            dpg.configure_item("manual_toggle", pos=자리)
            self._토글_자리 = 자리

    def _모드를_바꾼다(self, sender=None, app_data=None) -> None:
        self.sim.set_mode(next(m for m, 이름 in 모드_이름.items() if 이름 == app_data))
        self._알린다(f"카메라 입력 · {app_data}")

    def _모드를_세운다(self, mode: str) -> None:
        """모드를 바꾸고 라디오에도 되비춘다."""
        import dearpygui.dearpygui as dpg

        self.sim.set_mode(mode)
        dpg.set_value("verify_mode", 모드_이름[mode])

    @staticmethod
    def _영상_이름(영상) -> str:
        """한 줄: label · clip_id"""
        return f"{영상.label} · {영상.clip_id}"

    def _영상을_고른다(self, sender=None, app_data=None) -> None:
        import dearpygui.dearpygui as dpg

        이름들 = [self._영상_이름(영상) for 영상 in self._영상들]
        if str(app_data) not in 이름들:
            return
        self._고른_영상 = 이름들.index(str(app_data))
        dpg.set_value("video_text", self._영상들[self._고른_영상].text)

    def _영상_짝을_연다(self, loop: bool = True):
        """고른 영상의 프레임과 소리를 연다. 못 열면 (None, None) 이다."""
        클립 = self._영상들[self._고른_영상]
        try:
            그림 = sources.VideoFile.from_path(
                클립.video, size=self.sim.views.robot_size, loop=loop
            )
        except Exception as 오류:
            self._알린다(f"영상 열기 실패: {오류}")
            그림 = None

        try:
            소리 = sources.WavFile.from_path(클립.wav)
        except Exception as 오류:
            self._알린다(f"소리 열기 실패: {오류}")
            소리 = None

        return 그림, 소리

    def _영상_그림만_튼다(self, sender=None, app_data=None, user_data=None) -> None:
        """감정 표현 모드로 바꾸고 그 클립 영상을 되감아 반복으로 흘린다."""
        if not self._영상들:
            return
        클립 = self._영상들[self._고른_영상]
        try:
            그림 = sources.VideoFile.from_path(
                클립.video, size=self.sim.views.robot_size, loop=True
            )
        except Exception as 오류:
            self._알린다(f"영상 열기 실패: {오류}")
            return

        그림.rewind()
        self._모드를_세운다(모드_감정)
        self.sim.injector.play(t=self.sim.t, frames=그림)
        self._알린다(f"{클립.label} · 영상만")

    def _영상_소리와_그림_튼다(self, sender=None, app_data=None, user_data=None) -> None:
        """감정 표현 모드로 바꾸고 영상(반복)과 wav 를 같은 시각에서 함께 시작한다."""
        if not self._영상들:
            return
        그림, 소리 = self._영상_짝을_연다(loop=True)
        if 그림 is None or 소리 is None:
            return

        그림.rewind()
        self._모드를_세운다(모드_감정)
        self.sim.injector.play(t=self.sim.t, frames=그림, audio=소리)
        클립 = self._영상들[self._고른_영상]
        self._알린다(f"{클립.label} · 영상·소리 {소리.duration_s:.1f} s")

    def _TTS를_읽는다(self, sender=None, app_data=None, user_data=None) -> None:
        """적은 말을 내장 음성으로 읽어 마이크에 흘려 넣는다."""
        import dearpygui.dearpygui as dpg

        글 = dpg.get_value("tts_text")
        if not 글 or not 글.strip():
            self._알린다("문장 없음")
            return

        음성 = dpg.get_value("tts_voice")

        # **고리 밖에서 만든다.** `say` 는 1 초 가까이 걸린다. 고리 안에서 만들면
        # 그동안 온보드 SW 가 한 번도 불리지 않고 세상만 흐른다.
        def 만든다():
            try:
                소리 = sources.tts(글, voice=음성)
            except Exception as 오류:
                self._알린다(str(오류))
                return
            with self._소리_자물쇠:
                self._만든_소리.append(소리)
            self._알린다(f"말하는 중 · {소리.duration_s:.1f} s")

        threading.Thread(target=만든다, name="TTS", daemon=True).start()
        self._알린다("음성 만드는 중")

    def _고른_마이크(self):
        """고른 마이크의 번호와 이름이다. 목록이 없으면 기본 장치를 쓴다."""
        import dearpygui.dearpygui as dpg

        if not self._마이크_장치:
            return None, "기본 장치"
        고른_이름 = dpg.get_value("mic_device")
        for 번호, 이름 in self._마이크_장치:
            if 이름 == 고른_이름:
                return 번호, 이름
        return None, "기본 장치"

    def _마이크를_켠다(self, sender=None, app_data=None, user_data=None) -> None:
        번호, 이름 = self._고른_마이크()
        try:
            마이크 = sources.Mic(device=번호)
        except Exception as 오류:
            self._알린다(str(오류))
            return

        self.sim.injector.play(t=self.sim.t, audio=마이크)
        self._알린다(f"{이름} 입력 중")

    def _카메라를_켠다(self, sender=None, app_data=None, user_data=None) -> None:
        """맥북 카메라를 로봇 카메라 자리에 넣는다. 감정 표현 모드로 바꾼다."""
        import dearpygui.dearpygui as dpg

        번호 = int(dpg.get_value("cam_index"))
        try:
            캠 = sources.Webcam(index=번호, size=self.sim.views.robot_size)
        except Exception as 오류:
            self._알린다(str(오류))
            return

        self._모드를_세운다(모드_감정)
        self.sim.injector.play(t=self.sim.t, frames=캠)
        self._알린다(f"카메라 {번호} 입력 중")

    def _주입을_멈춘다(self, sender=None, app_data=None, user_data=None) -> None:
        self.sim.injector.stop()
        self.sim.frame = self.sim._검은_프레임
        self._알린다("입력 멈춤")

    def _구경꾼을_잡는다(self, sender=None, app_data=None) -> None:
        """그림 위에서 누른 것만 잡는다.

        누른 뒤에는 밖으로 나가도 계속 끌린다. 가장자리에서 손이 벗어났다고
        회전이 끊기면 쓰기 나쁘다.
        """
        if not self._구경꾼_잡음 and self._그림_위인가():
            self._구경꾼_잡음 = True
            self._끌기_앞 = (0.0, 0.0)

    def _구경꾼을_놓는다(self, sender=None, app_data=None) -> None:
        self._구경꾼_잡음 = False
        self._끌기_앞 = (0.0, 0.0)

    def _구경꾼을_돌린다(self, sender, app_data) -> None:
        if not self._구경꾼_잡음:
            return
        _, dx, dy = app_data
        앞_x, 앞_y = self._끌기_앞
        self.sim.views.camera.drag(dx - 앞_x, dy - 앞_y)
        self._끌기_앞 = (float(dx), float(dy))

    def _구경꾼을_당긴다(self, sender, app_data) -> None:
        if not self._그림_위인가():
            return
        self.sim.views.camera.zoom(float(app_data))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="AI Rookie 시뮬레이터")
    지난_정책 = 정책_폴더 / str(지난_입력().get("policy", ""))
    ap.add_argument(
        "--policy", type=Path, default=지난_정책 if 지난_정책.is_file() else 기본_정책,
        help=f"걷기 정책 ONNX. 기본은 지난번에 고른 것, 없으면 {기본_정책.name}. 화면에서도 바꿀 수 있다",
    )
    ap.add_argument("--real", action="store_true", help="실물로 켠다. 화면에서도 오갈 수 있다 (사양 4-4)")
    ap.add_argument("--host", default="jetson", help="--real 때 붙을 ssh 별칭")
    args = ap.parse_args()
    if args.real:
        App(real=True, host=args.host).run()
    else:
        App(Sim(policy=args.policy), host=args.host).run()


if __name__ == "__main__":
    main()
