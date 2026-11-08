"""한 주기 고리의 시험.

창을 띄우지 아니한다. 화면은 눈으로 봐야 하지만 고리는 그럴 필요가 없다.
"""

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from bridge import Bridge
from app import Sim, 기본_정책, 제어_주기_s, 카메라_fps, 카메라_주기_수
from inject import 모드_감정, 모드_추종
from sources import VideoFile, WavFile
from runtime import load_home_pose


@pytest.fixture
def 시뮬():
    # 같은 본문과 통로를 스레드에 담는다. 프로세스로 띄우는 것은 따로 본다.
    s = Sim(container="thread")
    yield s
    s.close()


def test_백_주기를_돌려도_서_있다(시뮬):
    for _ in range(100):
        시뮬.cycle()
    assert 시뮬.world.torso_height() > 0.10
    assert 시뮬.t == pytest.approx(100 * 제어_주기_s)


def test_프레임은_카메라_박자로만_찍힌다(시뮬):
    """제어 주기마다 찍으면 4K 에서 예산을 넘는다."""
    찍힌_주기 = 0
    for _ in range(100):
        앞 = 시뮬.frame
        시뮬.cycle()
        if 시뮬.frame is not 앞:
            찍힌_주기 += 1

    # 주기를 세므로 어림이 아니라 딱 떨어진다. 시각을 빼서 견주면 오차가
    # 쌓여 프레임이 조용히 빠진다.
    assert 카메라_주기_수 == 2
    assert 찍힌_주기 == 100 // 카메라_주기_수 == int(100 * 제어_주기_s * 카메라_fps)


# --- 검증 모드 (사양 3-3) ---


def 센서를_받아_적는다(시뮬) -> list:
    """주기마다 온보드 SW 로 넘어간 것을 모은다. 카메라 · 마이크는 지각 입구에 놓인 것이다."""
    받은_것, 이번 = [], {}

    def 감싼다(입구, 칸):
        원래_놓기 = 입구.put

        def 놓기(t, 값):
            이번[칸] = 값
            return 원래_놓기(t, 값)

        입구.put = 놓기

    감싼다(시뮬.camera_in, "frame")
    감싼다(시뮬.mic_in, "audio")
    원래 = 시뮬.runtime.step

    def 대신(sensors):
        받은_것.append(SimpleNamespace(
            t=sensors.t, sensors=sensors, frame=이번.pop("frame", None), audio=이번.pop("audio", None),
        ))
        return 원래(sensors)

    시뮬.runtime.step = 대신
    return 받은_것


def 가짜_얼굴(시뮬, 개수: int = 4, duration_s: float = 1.0) -> VideoFile:
    너비, 높이 = 시뮬.views.robot_size
    프레임들 = [
        np.full((높이, 너비, 3), 10 * (i + 1), dtype=np.uint8) for i in range(개수)
    ]
    return VideoFile(프레임들, fps=개수 / duration_s, loop=False)


def test_모드를_바꾸면_로봇_카메라를_비운다(시뮬):
    """앞 모드의 마지막 그림이 남으면 모드가 바뀐 것인지 가릴 수 없다."""
    시뮬.cycle()
    assert 시뮬.frame is not None
    assert 시뮬.frame.any()

    시뮬.set_mode(모드_감정)
    assert not 시뮬.frame.any()


def test_감정_모드에서는_월드를_렌더하지_아니한다(시뮬):
    """로봇 카메라에 들어가는 것이 주입한 그림이므로 렌더는 버릴 프레임이다."""
    센_것 = []
    원래 = 시뮬.views.robot_frame
    시뮬.views.robot_frame = lambda data, *a: (센_것.append(1), 원래(data, *a))[1]

    for _ in range(10):
        시뮬.cycle()
    렌더한_횟수 = len(센_것)
    assert 렌더한_횟수 > 0

    시뮬.set_mode(모드_감정)
    for _ in range(10):
        시뮬.cycle()
    assert len(센_것) == 렌더한_횟수


def test_감정_모드에서_주입한_얼굴이_센서_묶음에_실린다(시뮬):
    받은_것 = 센서를_받아_적는다(시뮬)
    시뮬.set_mode(모드_감정)
    시뮬.injector.play(t=시뮬.t, frames=가짜_얼굴(시뮬))

    for _ in range(10):
        시뮬.cycle()

    프레임들 = [s.frame for s in 받은_것 if s.frame is not None]
    assert 프레임들
    assert 프레임들[0][0, 0, 0] == 10


def test_추종_모드로_돌아오면_다시_월드를_렌더한다(시뮬):
    시뮬.set_mode(모드_감정)
    시뮬.cycle()

    시뮬.set_mode(모드_추종)
    for _ in range(카메라_주기_수):
        시뮬.cycle()
    assert 시뮬.frame is not None
    assert 시뮬.frame.any()


def test_소리는_모드를_가리지_아니한다(시뮬):
    """추종 모드에서도 호출어를 넣어야 한다."""
    받은_것 = 센서를_받아_적는다(시뮬)
    시뮬.injector.play(t=시뮬.t, audio=WavFile(np.ones(16000, dtype=np.float32), 16000))

    assert 시뮬.injector.mode == 모드_추종
    for _ in range(5):
        시뮬.cycle()

    소리들 = [s.audio for s in 받은_것 if s.audio is not None]
    assert len(소리들) == 5
    # 마이크 자리는 int16 48 kHz 다 (계약 2-3). 20 ms 는 960 표본이다.
    assert all(len(블록) == 960 and 블록.dtype == np.int16 for 블록 in 소리들)


def test_소리가_다_나가면_무음이_실린다(시뮬):
    """로봇 마이크는 늘 듣는다. 비우면 발화 뒤 무음이 안 와 발화가 안 닫힌다."""
    받은_것 = 센서를_받아_적는다(시뮬)
    # 두 주기 분량만 넣는다.
    시뮬.injector.play(t=시뮬.t, audio=WavFile(np.ones(640, dtype=np.float32), 16000))

    for _ in range(5):
        시뮬.cycle()

    assert all(s.audio is not None and len(s.audio) == 960 for s in 받은_것)
    assert [bool(s.audio.any()) for s in 받은_것] == [True, True, False, False, False]


def test_틀_것이_없어도_무음이_실린다(시뮬):
    받은_것 = 센서를_받아_적는다(시뮬)
    시뮬.cycle(3)
    assert len(받은_것[0].audio) == 3 * 960 and not 받은_것[0].audio.any()
    assert 받은_것[0].audio.dtype == np.int16


def test_TTS_는_고리_밖에서_만들고_고리가_튼다(시뮬):
    """`say` 가 1 초 가까이 걸린다. 고리 안에서 만들면 온보드 SW 가 그동안 안 불린다."""
    from app import App

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    소리 = WavFile(np.ones(1600, dtype=np.float32), 16000)
    with 앱._소리_자물쇠:
        앱._만든_소리.append(소리)
    앱._한_바퀴(가짜)
    assert 시뮬.injector.playing


def test_온보드_SW_는_앱이_부를_때만_돈다(시뮬):
    """시계는 앱이 쥔다. 안 부르면 아무 일도 일어나지 아니한다."""
    시뮬.cycle()
    앞 = 시뮬.runtime.snapshot().t
    높이 = 시뮬.world.torso_height()
    assert 시뮬.runtime.snapshot().t == 앞
    assert 시뮬.world.torso_height() == 높이


def test_주입한_전압이_온보드_SW_까지_간다(시뮬):
    시뮬.injector.drop_voltage(6.2)
    시뮬.cycle()
    np.testing.assert_allclose(시뮬.runtime.snapshot().servo.voltage, np.full(14, 6.2))


def test_한_주기에_든_시간을_잰다(시뮬):
    시뮬.cycle()
    assert 시뮬.cycle_ms > 0.0


def test_조회가_열일곱_덩어리다(시뮬):
    """서보와 IMU, 그리고 계층 다섯이다."""
    assert 시뮬.status() == ()
    시뮬.cycle()
    assert len(시뮬.status()) == 7


class 빈_문맥:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class 가짜_dpg:
    """창 대신 값을 받아 두기만 한다. 시계 의미는 창 없이 시험된다."""

    mvTable_SizingFixedFit = 0

    def __init__(self):
        self.값 = {}
        self.글자 = []
        self.지운_횟수 = 0
        self.표_지은_횟수 = 0

    def set_value(self, tag, value):
        self.값[tag] = value

    def get_value(self, tag):
        return self.값.get(tag)

    def delete_item(self, tag, children_only=False):
        self.지운_횟수 += 1
        self.글자.clear()

    def add_text(self, text, parent=None, tag=None):
        self.글자.append(text)
        if tag is not None:
            self.값[tag] = text

    def table(self, **kwargs):
        self.표_지은_횟수 += 1
        return 빈_문맥()

    def table_row(self):
        return 빈_문맥()

    def add_table_column(self, label=""):
        self.글자.append(label)

    def add_spacer(self, height=0, parent=None, tag=None):
        pass

    def add_group(self, tag=None):
        return 빈_문맥()

    def set_y_scroll(self, tag, value):
        pass

    def get_y_scroll_max(self, tag):
        return 0.0

    def configure_item(self, tag, **kwargs):
        self.값[f"cfg:{tag}"] = kwargs

    def get_item_pos(self, tag):
        return (0, 0)

    def get_item_rect_size(self, tag):
        return (0, 0)


def 시계앱(시뮬):
    """시계를 갈아 끼운 앱을 낸다. 벽시계를 손으로 민다."""
    from app import App

    앱 = App(시뮬)
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]
    return 앱, 가짜_dpg(), 시각


def test_벽시계_20_ms_마다_한_주기가_돈다(시뮬):
    앱, 가짜, 시각 = 시계앱(시뮬)
    앱._한_바퀴(가짜)
    assert 시뮬.cycles == 1

    시각[0] = 0.019
    앱._한_바퀴(가짜)
    assert 시뮬.cycles == 1

    시각[0] = 0.02
    앱._한_바퀴(가짜)
    assert 시뮬.cycles == 2
    assert 시뮬.missed == 0


def test_늦으면_빠진_주기만큼_세상이_먼저_흐른다(시뮬):
    """물리는 따라잡고 온보드 SW 는 최신 센서로 한 번만 부른다 (사양 1-2)."""
    앱, 가짜, 시각 = 시계앱(시뮬)
    받은_것 = 센서를_받아_적는다(시뮬)
    앱._한_바퀴(가짜)

    시각[0] = 0.1
    앱._한_바퀴(가짜)

    assert 시뮬.cycles == 6
    assert 시뮬.missed == 4
    assert len(받은_것) == 2
    # 온보드 SW 가 받은 시각이 벽시계와 같다. 20 ms 씩 오른다고 가정하면 안 된다.
    assert 받은_것[-1].t == pytest.approx(0.1)
    # 물리도 벽시계만큼 흘렀다. 마지막 주기가 한 칸 더 민다.
    assert 시뮬.world.data.time == pytest.approx(0.12)


def test_빠진_주기의_소리도_버리지_아니한다(시뮬):
    """마이크는 빠진 주기에도 들었다."""
    받은_것 = 센서를_받아_적는다(시뮬)
    시뮬.injector.play(t=시뮬.t, audio=WavFile(np.ones(16000, dtype=np.float32), 16000))

    시뮬.cycle(3)
    assert len(받은_것[0].audio) == 3 * 960


def test_주기를_건너뛰어도_카메라_박자를_넘겼으면_찍는다(시뮬):
    """나머지가 0 인 주기를 기다리면 그 주기를 건너뛸 때 프레임이 빠진다."""
    받은_것 = 센서를_받아_적는다(시뮬)
    시뮬.cycle()  # 0 번 주기에 찍는다
    시뮬.cycle()  # 1 번은 박자 안이다

    # 2 번을 건너뛰고 3 번에 부른다
    시뮬.cycle(2)
    assert [s.frame is not None for s in 받은_것] == [True, False, True]


def test_주기_수가_1_보다_작으면_터뜨린다(시뮬):
    with pytest.raises(ValueError):
        시뮬.cycle(0)


def test_초기화하면_처음으로_돌아간다(시뮬):
    앞_런타임 = 시뮬.runtime
    for _ in range(5):
        시뮬.cycle()
    시뮬.cycle(3)
    시뮬.world.set_person_pos(1.0, 1.0)
    시뮬.injector.drop_voltage(6.2)
    시뮬.injector.play(t=시뮬.t, audio=WavFile(np.ones(16000, dtype=np.float32), 16000))

    시뮬.reset()

    assert (시뮬.t, 시뮬.cycles, 시뮬.missed) == (0.0, 0, 0)
    assert 시뮬.world.data.time == 0.0
    assert 시뮬.world.person_pos == pytest.approx((5.0, 0.0))
    # 온보드 SW 는 새로 만든다. 한 주기도 안 돌았으니 조회가 없다.
    assert 시뮬.runtime is not 앞_런타임
    assert 시뮬.runtime.snapshot() is None
    assert not 시뮬.injector.playing

    시뮬.cycle()
    전압 = 시뮬.runtime.snapshot().servo.voltage
    # 주입한 6.2 V 가 지워지고 하중으로 계산한 전압이 실린다 (사양 2-4).
    assert np.all(전압 > 7.0) and np.all(전압 <= Bridge.기본_전압_V)


def test_초기화는_검증_모드를_두고_간다(시뮬):
    시뮬.set_mode(모드_감정)
    시뮬.reset()
    assert 시뮬.injector.mode == 모드_감정


def test_초기화한_뒤에는_벽시계를_다시_센다(시뮬):
    앱, 가짜, 시각 = 시계앱(시뮬)
    앱._한_바퀴(가짜)
    시각[0] = 5.0
    앱._한_바퀴(가짜)

    시뮬.reset()
    앱._시작 = None
    앱._한_바퀴(가짜)
    assert (시뮬.cycles, 시뮬.missed) == (1, 0)


def test_빠진_주기를_화면에_띄운다(시뮬):
    앱, 가짜, 시각 = 시계앱(시뮬)
    앱._한_바퀴(가짜)
    assert "늦은 주기 0" in 가짜.값["clock"]


def test_한_주기_시간을_화면에_띄운다(시뮬):
    from app import App

    가짜 = 가짜_dpg()
    App(시뮬)._한_바퀴(가짜)
    assert "ms" in 가짜.값["clock"] and "한도" in 가짜.값["clock"]


def test_글자는_늦춰서_갱신한다(시뮬):
    """60 Hz 로 바뀌는 숫자는 읽을 수가 없다. 그림은 매 프레임 갱신한다."""
    from app import App, 글자_갱신_주기_s

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]

    앱._한_바퀴(가짜)
    첫_글자 = 가짜.값["clock"]
    첫_그림 = 가짜.값["spectator_tex"]

    가짜.값.clear()
    시각[0] = 글자_갱신_주기_s * 0.5
    앱._한_바퀴(가짜)
    assert "clock" not in 가짜.값
    assert "spectator_tex" in 가짜.값

    시각[0] = 글자_갱신_주기_s * 1.5
    앱._한_바퀴(가짜)
    assert 가짜.값["clock"] != 첫_글자


def test_조회를_표로_그린다(시뮬):
    """공백으로 칸을 맞추면 글꼴 너비가 달라 어긋난다."""
    from app import App

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    앱._한_바퀴(가짜)

    # 머리줄과 관절 이름이 표의 칸으로 들어간다
    assert "관절" in 가짜.글자 and "실제 (rad)" in 가짜.글자
    assert "left_hip_yaw" in 가짜.글자
    # 공백으로 맞춘 한 덩어리 글자를 만들지 아니한다
    assert not any("  " in 글 for 글 in 가짜.글자 if isinstance(글, str))


def test_한_주기_시간을_모아_평균과_최대로_낸다(시뮬):
    """늦춰 놓고 직전 하나만 띄우면 어느 주기가 걸렸느냐에 따라 널뛴다."""
    from app import App, 글자_갱신_주기_s

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]
    앱._한_바퀴(가짜)

    for _ in range(5):
        시각[0] += 글자_갱신_주기_s * 0.1
        앱._한_바퀴(가짜)
    assert len(앱._주기_시간) == 5

    시각[0] += 글자_갱신_주기_s
    앱._한_바퀴(가짜)
    글자 = 가짜.값["clock"]
    assert "평균" in 글자 and "최대" in 글자
    assert 앱._주기_시간 == []


def test_모양이_그대로면_표를_다시_만들지_아니한다(시뮬):
    """다시 만든 표는 한 프레임 동안 열이 찌그러진다. 그게 깜박임이다."""
    from app import App, 글자_갱신_주기_s

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]

    앱._한_바퀴(가짜)
    지은_횟수 = 가짜.표_지은_횟수
    assert 지은_횟수 > 0

    for _ in range(5):
        시각[0] += 글자_갱신_주기_s * 1.1
        앱._한_바퀴(가짜)
    assert 가짜.표_지은_횟수 == 지은_횟수


def test_칸_값은_계속_갈아_끼운다(시뮬):
    from app import App, 글자_갱신_주기_s

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]
    앱._한_바퀴(가짜)

    # 서보 표는 계층 상세 뒤에 있다 (`App._차례`). 제목으로 자리를 찾는다.
    서보 = next(i for i, d in enumerate(앱._차례(시뮬.status())) if d.title == "서보")
    칸 = f"cell_{서보}_0_1"
    첫_값 = 가짜.값[칸]
    시뮬.world.data.qpos[시뮬.world.model.jnt_qposadr[
        시뮬.world.model.actuator_trnid[0, 0]
    ]] += 0.3
    시각[0] += 글자_갱신_주기_s * 1.1
    앱._한_바퀴(가짜)
    assert 가짜.값[칸] != 첫_값


def test_계층이_서면_그때는_다시_만든다(시뮬):
    """줄 수가 바뀌면 칸만 갈아 끼울 수가 없다."""
    from dataclasses import dataclass

    from app import App, 글자_갱신_주기_s
    from panels import Section

    @dataclass
    class 안전조회:
        state: str
        voltage_min: float

    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]
    앱._한_바퀴(가짜)
    지은_횟수 = 가짜.표_지은_횟수

    원래 = 시뮬.status
    시뮬.status = lambda: tuple(
        Section(title=d.title, header=d.header,
                rows=(("state", "정상"), ("voltage_min", "7.1")))
        if d.title == "안전" else d
        for d in 원래()
    )
    시각[0] += 글자_갱신_주기_s * 1.1
    앱._한_바퀴(가짜)
    assert 가짜.표_지은_횟수 > 지은_횟수


def test_그림_밖에서_끌면_시점이_안_움직인다(시뮬):
    from app import App

    앱 = App(시뮬)
    앱._그림_위인가 = lambda: False
    앞 = 시뮬.views.camera.azimuth

    앱._구경꾼을_잡는다()
    앱._구경꾼을_돌린다(None, (0, 100.0, 0.0))
    assert 시뮬.views.camera.azimuth == 앞


def test_그림_위에서_누르면_끌린다(시뮬):
    from app import App

    앱 = App(시뮬)
    앱._그림_위인가 = lambda: True
    앞 = 시뮬.views.camera.azimuth

    앱._구경꾼을_잡는다()
    앱._구경꾼을_돌린다(None, (0, 100.0, 0.0))
    assert 시뮬.views.camera.azimuth != 앞


def test_끌기는_누적이라_차이만_쓴다(시뮬):
    """dearpygui 가 누른 지점부터의 누적 이동량을 준다.

    그대로 더하면 끄는 동안 회전이 점점 빨라진다.
    """
    from app import App

    앱 = App(시뮬)
    앱._그림_위인가 = lambda: True
    앱._구경꾼을_잡는다()

    앱._구경꾼을_돌린다(None, (0, 10.0, 0.0))
    한_번 = 시뮬.views.camera.azimuth
    앱._구경꾼을_돌린다(None, (0, 10.0, 0.0))
    assert 시뮬.views.camera.azimuth == 한_번

    앱._구경꾼을_돌린다(None, (0, 20.0, 0.0))
    두_번 = 시뮬.views.camera.azimuth
    assert 두_번 != 한_번
    # 10 픽셀씩 두 번 움직인 것과 20 픽셀 한 번이 같아야 한다
    앱2 = App(시뮬)
    앱2._그림_위인가 = lambda: True
    시뮬.views.camera.azimuth = 0.0
    앱2._구경꾼을_잡는다()
    앱2._구경꾼을_돌린다(None, (0, 20.0, 0.0))
    한_번에 = 시뮬.views.camera.azimuth

    시뮬.views.camera.azimuth = 0.0
    앱3 = App(시뮬)
    앱3._그림_위인가 = lambda: True
    앱3._구경꾼을_잡는다()
    앱3._구경꾼을_돌린다(None, (0, 10.0, 0.0))
    앱3._구경꾼을_돌린다(None, (0, 20.0, 0.0))
    assert 시뮬.views.camera.azimuth == pytest.approx(한_번에)


def test_놓으면_다음_끌기가_처음부터다(시뮬):
    from app import App

    앱 = App(시뮬)
    앱._그림_위인가 = lambda: True
    앱._구경꾼을_잡는다()
    앱._구경꾼을_돌린다(None, (0, 50.0, 0.0))
    앱._구경꾼을_놓는다()

    앱._그림_위인가 = lambda: False
    앱._구경꾼을_잡는다()
    앞 = 시뮬.views.camera.azimuth
    앱._구경꾼을_돌린다(None, (0, 50.0, 0.0))
    assert 시뮬.views.camera.azimuth == 앞


def test_휠은_그림_위에서만_듣는다(시뮬):
    from app import App

    앱 = App(시뮬)
    앱._그림_위인가 = lambda: False
    앞 = 시뮬.views.camera.distance
    앱._구경꾼을_당긴다(None, 3.0)
    assert 시뮬.views.camera.distance == 앞

    앱._그림_위인가 = lambda: True
    앱._구경꾼을_당긴다(None, 3.0)
    assert 시뮬.views.camera.distance != 앞


def test_사람이_기본으로_로봇을_바라본다(시뮬):
    import math

    사람_각 = 시뮬.world.person_yaw
    로봇을_보는_각 = 시뮬.world.로봇을_바라보는_각()
    assert math.isclose(사람_각, 로봇을_보는_각, abs_tol=1e-6)


def test_사람을_돌릴_수_있다(시뮬):
    import math

    시뮬.world.set_person_yaw(math.radians(45.0))
    assert math.degrees(시뮬.world.person_yaw) == pytest.approx(45.0)


def test_사람을_옮겨도_방향은_그대로다(시뮬):
    """옮길 때마다 제멋대로 돌면 슬라이더와 싸운다."""
    import math

    시뮬.world.set_person_yaw(math.radians(90.0))
    시뮬.world.set_person_pos(2.0, -1.0)
    assert math.degrees(시뮬.world.person_yaw) == pytest.approx(90.0)


def test_기본_시점에_로봇과_사람이_둘_다_들어온다(시뮬):
    """로봇 뒤 정면(방위각 0)에서는 사람이 로봇에 가려 겹친다."""
    import numpy as np

    # 외장 색은 바닥 회색과 가려지지 않으므로 로봇 겉모습만 순수 빨강으로 칠해 찾는다.
    model = 시뮬.world.model
    뿌리 = model.body_rootid[model.body("base").id]
    로봇 = np.nonzero((model.geom_group == 2) & (model.body_rootid[model.geom_bodyid] == 뿌리))[0]
    원래 = model.geom_rgba[로봇].copy(), model.geom_matid[로봇].copy()
    model.geom_rgba[로봇], model.geom_matid[로봇] = [1.0, 0.0, 0.0, 1.0], -1
    try:
        f = 시뮬.views.spectator_frame(시뮬.world.data)
    finally:
        model.geom_rgba[로봇], model.geom_matid[로봇] = 원래
    h, w, _ = f.shape
    r, g, b = f[:, :, 0].astype(int), f[:, :, 1].astype(int), f[:, :, 2].astype(int)

    빨강 = np.nonzero((r > 120) & (g < 40) & (b < 40))
    살색 = np.nonzero((r - b > 35) & (r > 150) & (g > 120))
    assert len(빨강[0]) > 0, "로봇이 화면에 없다"
    assert len(살색[0]) > 300, "사람이 화면에 없다"

    로봇_가로 = 빨강[1].mean() / w
    사람_가로 = 살색[1].mean() / w
    assert abs(로봇_가로 - 사람_가로) > 0.1, "둘이 겹쳐 보인다"


def 키앱(시뮬, 눌린=()):
    """키 판정과 시계를 갈아 끼운 앱을 낸다."""
    from app import App

    앱 = App(시뮬)
    시각 = [0.0]
    앱._시계 = lambda: 시각[0]
    앱._키가_눌렸나 = lambda 이름: 이름 in 눌린
    return 앱, 가짜_dpg(), 시각


def test_W_는_바라보는_쪽으로_걷는다(시뮬):
    from app import 사람_속도_mps

    시뮬.world.set_person_pos(3.0, 0.0)
    시뮬.world.set_person_yaw(0.0)  # +x 를 본다
    앱, 가짜, 시각 = 키앱(시뮬, {"W"})

    앱._사람을_키로_움직인다(가짜)  # 첫 호출은 시각만 잡는다
    시각[0] = 0.05
    앱._사람을_키로_움직인다(가짜)

    x, y = 시뮬.world.person_pos
    assert x == pytest.approx(3.0 + 사람_속도_mps * 0.05)
    assert y == pytest.approx(0.0)


def test_돌려_놓으면_W_가_그_쪽으로_간다(시뮬):
    """월드 축으로 움직이면 사람을 돌린 뒤 W 가 엉뚱한 쪽으로 간다."""
    import math

    from app import 사람_속도_mps

    시뮬.world.set_person_pos(0.0, 0.0)
    시뮬.world.set_person_yaw(math.radians(90.0))  # +y 를 본다
    앱, 가짜, 시각 = 키앱(시뮬, {"W"})

    앱._사람을_키로_움직인다(가짜)
    시각[0] = 0.05
    앱._사람을_키로_움직인다(가짜)

    x, y = 시뮬.world.person_pos
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(사람_속도_mps * 0.05)


def test_A_와_D_는_옆으로_간다(시뮬):
    from app import 사람_속도_mps

    시뮬.world.set_person_pos(0.0, 0.0)
    시뮬.world.set_person_yaw(0.0)
    앱, 가짜, 시각 = 키앱(시뮬, {"D"})

    앱._사람을_키로_움직인다(가짜)
    시각[0] = 0.05
    앱._사람을_키로_움직인다(가짜)

    x, y = 시뮬.world.person_pos
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(-사람_속도_mps * 0.05)


def test_Q_와_E_는_돌린다(시뮬):
    import math

    from app import 사람_회전_dps

    시뮬.world.set_person_yaw(0.0)
    앱, 가짜, 시각 = 키앱(시뮬, {"Q"})

    앱._사람을_키로_움직인다(가짜)
    시각[0] = 0.05
    앱._사람을_키로_움직인다(가짜)

    assert math.degrees(시뮬.world.person_yaw) == pytest.approx(사람_회전_dps * 0.05)


def test_안_누르면_안_움직인다(시뮬):
    시뮬.world.set_person_pos(1.0, 2.0)
    앱, 가짜, 시각 = 키앱(시뮬, set())

    앱._사람을_키로_움직인다(가짜)
    시각[0] = 0.05
    앱._사람을_키로_움직인다(가짜)

    assert 시뮬.world.person_pos == pytest.approx((1.0, 2.0))


def test_창이_멈췄다_돌아와도_한_번에_안_튄다(시뮬):
    """벽시계라 창이 오래 멈춰 있으면 dt 가 통째로 밀려 들어온다."""
    from app import 사람_속도_mps

    시뮬.world.set_person_pos(0.0, 0.0)
    시뮬.world.set_person_yaw(0.0)
    앱, 가짜, 시각 = 키앱(시뮬, {"W"})

    앱._사람을_키로_움직인다(가짜)
    시각[0] = 30.0
    앱._사람을_키로_움직인다(가짜)

    x, _ = 시뮬.world.person_pos
    assert x == pytest.approx(사람_속도_mps * 0.1)


def test_사람은_제어_주기가_아니라_창_박자로_움직인다(시뮬):
    """월드 조작은 온보드 SW 를 거치지 아니한다 (사양 3-4)."""
    시뮬.world.set_person_pos(0.0, 0.0)
    시뮬.world.set_person_yaw(0.0)
    앱, 가짜, 시각 = 키앱(시뮬, {"W"})

    앱._한_바퀴(가짜)
    시각[0] = 0.05
    앱._한_바퀴(가짜)

    assert 시뮬.world.person_pos[0] > 0.0


# --- 걷기 정책 (사양 1-3) ---


def test_기본_정책은_v2_48_이다(시뮬):
    assert 시뮬.policy.name == "v2_48_300M.onnx"


def test_기본_정책은_하위_폴더에서_찾고_없으면_알린다(tmp_path):
    from app import 기본으로_고를_정책

    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / "v2_48_300M.onnx").touch()
    (tmp_path / "v2" / "v2_53s1_300M.onnx").touch()
    assert 기본으로_고를_정책(tmp_path) == tmp_path / "v2" / "v2_48_300M.onnx"
    (tmp_path / "v2" / "v2_48_300M.onnx").unlink()
    with pytest.raises(FileNotFoundError):
        기본으로_고를_정책(tmp_path)



def test_정책을_바꾸면_처음부터_돈다(시뮬):
    from app import 정책_폴더

    for _ in range(5):
        시뮬.cycle()
    앞_런타임 = 시뮬.runtime

    시뮬.set_policy(정책_폴더 / "v1" / "v1_22_300M.onnx")

    assert 시뮬.policy.name == "v1_22_300M.onnx"
    assert 시뮬.runtime is not 앞_런타임
    assert (시뮬.t, 시뮬.cycles) == (0.0, 0)


def test_못_여는_정책이면_앞_정책이_그대로_남는다(시뮬):
    앞_런타임 = 시뮬.runtime
    with pytest.raises(Exception):
        시뮬.set_policy("/없는/정책.onnx")
    assert 시뮬.policy == 기본_정책
    assert 시뮬.runtime is 앞_런타임


def test_고를_수_있는_정책을_이름_차례로_낸다():
    from app import 정책들

    이름들 = [p.name for p in 정책들()]
    assert "v1_23_300M.onnx" in 이름들
    assert 이름들 == sorted(이름들)


def test_박스는_화면에만_그리고_온보드가_받은_프레임은_그대로다(시뮬):
    """중앙 위는 온보드 SW 에 들어간 것과 같은 프레임이어야 한다 (사양 4-1)."""
    from app import App

    앱 = App(시뮬)
    너비, 높이 = 시뮬.views.robot_size
    원본 = np.full((높이, 너비, 3), 128, dtype=np.uint8)
    앞 = 원본.copy()
    버퍼 = 앱._빈_텍스처(너비, 높이)
    앱._텍스처에_넣는다(버퍼, 원본)
    앱._박스를_그린다(버퍼, 너비, 높이, ((1, 100, 50, 200, 300),))

    그림 = 버퍼.reshape(높이, 너비, 4)
    assert 그림[50, 150, 1] == 1.0   # 윗변은 초록이다
    assert 그림[150, 150, 1] == pytest.approx(128 / 255)  # 안쪽은 그대로다
    np.testing.assert_array_equal(원본, 앞)


def test_고른_대상은_노랑으로_그린다(시뮬):
    from app import App

    앱 = App(시뮬)
    너비, 높이 = 시뮬.views.robot_size
    버퍼 = 앱._빈_텍스처(너비, 높이)
    앱._박스를_그린다(버퍼, 너비, 높이, ((1, 100, 50, 200, 300), (2, 400, 50, 500, 300)), 고른_id=2)
    그림 = 버퍼.reshape(높이, 너비, 4)
    assert tuple(그림[50, 150, :3]) == pytest.approx((0.2, 1.0, 0.2))
    assert tuple(그림[50, 450, :3]) == pytest.approx((1.0, 0.85, 0.0))


def test_표현_검증_버튼이_온보드_재생기까지_간다(시뮬):
    """서 있는 채로 누르면 다음 주기부터 귀에 힘이 들어 드라이버 자리까지 간다 (사양 3-5)."""
    시뮬.cycle()
    시뮬.trigger_expression("call_heard")
    시뮬.cycle()
    assert 시뮬.bridge.ear[0] is not None
    assert 시뮬.runtime.snapshot().approach.expression.endswith("call_heard")


# --- 수동 조작 (Runtime 자율 사양 8장) ---


def test_수동_조작은_조종기_입력을_센서_묶음에_싣는다(시뮬):
    받은 = []
    원래 = 시뮬.runtime.step

    def 대신(묶음):
        받은.append(묶음.manual)
        return 원래(묶음)

    시뮬.runtime.step = 대신
    시뮬.cycle()
    시뮬.set_manual(True)
    시뮬.drive(forward=1.0, lateral=0.0, turn=-1.0)
    시뮬.cycle()
    assert 받은[0] is None
    assert (받은[1].forward, 받은[1].turn) == (1.0, -1.0)


def test_수동_조작_중_키는_사람_대신_로봇을_움직인다(시뮬):
    시뮬.world.set_person_pos(3.0, 0.0)
    시뮬.set_manual(True)
    앱, 가짜, 시각 = 키앱(시뮬, {"W", "A", "Q"})
    앱._사람을_키로_움직인다(가짜)
    시각[0] = 0.05
    앱._사람을_키로_움직인다(가짜)

    assert 시뮬.world.person_pos == pytest.approx((3.0, 0.0))
    조종 = 시뮬._조종
    # 로봇 조종기는 왼쪽(A)·왼쪽으로 돌기(Q)가 + 다
    assert (조종.forward, 조종.lateral, 조종.turn) == (1.0, 1.0, 1.0)


def test_수동_조작_중_화살표는_머리_ZC_는_귀다(시뮬):
    시뮬.set_manual(True)
    앱, 가짜, _ = 키앱(시뮬, {"↑", "←", "Z"})
    앱._사람을_키로_움직인다(가짜)
    조종 = 시뮬._조종
    assert (조종.head_pitch, 조종.head_yaw, 조종.head_home, 조종.ear) == (1.0, 1.0, False, (True, False))


def test_키를_놓으면_세_축이_0_이다(시뮬):
    시뮬.set_manual(True)
    앱, 가짜, _ = 키앱(시뮬, set())
    앱._사람을_키로_움직인다(가짜)
    조종 = 시뮬._조종
    assert (조종.forward, 조종.lateral, 조종.turn) == (0.0, 0.0, 0.0)


def test_수동_조작_토글은_구경꾼_그림_우측_상단에_있다(시뮬):
    from app import App, 토글_여백, 토글_폭

    class 그림_있는_dpg(가짜_dpg):
        def get_item_pos(self, tag):
            return (10, 300)

        def get_item_rect_size(self, tag):
            return (640, 320)

    앱 = App(시뮬)
    가짜 = 그림_있는_dpg()
    앱._토글을_얹는다(가짜)
    assert 가짜.값["cfg:manual_toggle"]["pos"] == [10 + 640 - 토글_폭 - 토글_여백, 300 + 토글_여백]


def test_두_시점은_세로가_같고_폭은_각자_비율이며_가운데에_선다(시뮬):
    """로봇 시점은 4:3, 구경꾼은 2:1 로 그린다. 한쪽 비율로 둘을 그리면 다른 쪽이 찌그러진다."""
    from app import App

    class 크기를_적는_dpg:
        def __init__(self):
            self.크기 = {}

        def configure_item(self, tag, **kw):
            if "width" in kw and "height" in kw:
                self.크기[tag] = (kw["width"], kw["height"])
            elif "width" in kw:
                self.크기[tag] = kw["width"]

    앱 = App(시뮬)
    가짜 = 크기를_적는_dpg()
    for 중앙_폭, 높이 in ((1400, 400), (700, 400)):
        앱._그림_높이 = 높이
        앱._그림에_맞춘다(가짜, 중앙_폭)
        (로_폭, 로_높이), (구_폭, 구_높이) = 가짜.크기["robot_img"], 가짜.크기["spectator_img"]
        assert 로_높이 == 구_높이
        assert 로_폭 / 로_높이 == pytest.approx(4 / 3, abs=0.01)
        assert 구_폭 / 구_높이 == pytest.approx(2.0, abs=0.01)
        assert 구_폭 <= 중앙_폭 - 24
        # 두 그림 모두 중앙 자리 가운데에 선다.
        for 태그, 폭 in (("robot", 로_폭), ("spectator", 구_폭)):
            여백 = 가짜.크기[f"{태그}_left_pad"]
            assert abs(여백 + 폭 / 2 - (중앙_폭 - 24) / 2) <= 1


def test_프로세스로_띄운_지각이_사람을_찾고_초기화해도_다시_띄우지_않는다():
    """실물과 같은 세 프로세스다. 시뮬 앱은 로봇 카메라 렌더를 지각 입구에 놓기만 한다."""
    import time
    from pathlib import Path

    모델 = Path(__file__).resolve().parents[2] / "AIRookie_Runtime" / "models" / "yolov8n.onnx"
    if not 모델.exists():
        pytest.skip("검출 모델이 없다")
    s = Sim()
    try:
        assert s.senses.container == "process"
        일꾼 = [p.pid for p in s.senses._일꾼]

        def 사람을_찾는다(제한_s=30.0) -> bool:
            끝 = time.monotonic() + 제한_s
            while time.monotonic() < 끝:
                s.cycle()
                time.sleep(제어_주기_s)
                조회 = s.runtime.snapshot()
                if 조회 is not None and 조회.perception.tracks != "없다":
                    return True
            return False

        assert 사람을_찾는다()
        s.reset()
        assert [p.pid for p in s.senses._일꾼] == 일꾼
        assert all(p.is_alive() for p in s.senses._일꾼)
        assert 사람을_찾는다()
    finally:
        s.close()
    assert not any(p.is_alive() for p in s.senses._일꾼)


def test_시뮬도_앱_명령을_실물과_같은_입구로_따른다(시뮬):
    """실물 디버깅 도구 · 수동 속도 (사양 4-4 · 4-5) 를 시뮬에서도 같은 Runtime 입구로 받는다."""
    시뮬.cycle()
    assert 시뮬.send_command({"kind": "expression", "name": "call_heard"})
    시뮬.cycle()
    assert 시뮬.runtime.snapshot().approach.expression.endswith("call_heard")
    assert 시뮬.send_command({"kind": "volume", "value": 0.4})
    assert 시뮬.runtime._volume == pytest.approx(0.4)
    assert 시뮬.send_command({"kind": "idle"})
    assert not 시뮬.send_command({"kind": "없는 명령"})


def test_처음부터_해도_수동_속도와_볼륨은_남는다(시뮬):
    시뮬.send_command({"kind": "volume", "value": 0.3})
    시뮬.send_command({"kind": "manual_speed", "vx_max": 0.1, "vx_back_max": 0.1, "vy_max": 0.05, "wz_max": 0.3})
    시뮬.reset()
    assert 시뮬.runtime._volume == pytest.approx(0.3)
    assert 시뮬.runtime._manual_back == pytest.approx(0.1)


def test_시뮬도_실행할_때_실물과_같은_설정을_건다(시뮬):
    """실물 실행 인자(--arm · --safety · --no-servo · --servo-speed)를 시뮬도 실행할 때 건다."""
    시뮬.launch(시뮬.policy, arm=False, safety=False, no_servo=True, servo_speed=300)
    assert 시뮬.running and 시뮬.no_servo and 시뮬.runtime.safety_on is False
    assert not 시뮬.bridge.armed and 시뮬.bridge.servo_speed == 300
    시뮬.launch(시뮬.policy, arm=True, safety=True, no_servo=False, servo_speed=0)
    assert 시뮬.bridge.armed and 시뮬.runtime.safety_on is True and not 시뮬.no_servo


def test_시뮬_정지는_온보드_SW_를_멈추고_홈_자세를_쥔다(시뮬):
    _, 홈 = load_home_pose()
    시뮬.stop()
    assert not 시뮬.running
    시뮬.cycle()
    assert np.allclose(시뮬.world.data.ctrl, 홈)


def test_시뮬도_최근_기록을_모은다(시뮬):
    시뮬.cycle()
    시뮬.status()
    assert 시뮬.history is not None


def test_맥_em7_을_고르면_실물은_경로의_맥_주소_시뮬은_이_맥으로_감정을_보낸다(시뮬, monkeypatch):
    import app as 앱_모듈
    from app import App

    monkeypatch.setattr(앱_모듈, "asr_url", lambda route, port: f"http://{route}:{port}")
    앱 = App(시뮬)
    가짜 = 가짜_dpg()
    가짜.값["emotion"] = "기본 감정"
    assert 앱._고른_감정(가짜, 실물=True) == ("", "")
    가짜.값["emotion"] = "맥 em7"
    앱._경로 = "NetBird"
    assert 앱._고른_감정(가짜, 실물=True) == ("mac", "http://NetBird:8766")
    assert 앱._고른_감정(가짜, 실물=False) == ("mac", "http://127.0.0.1:8766")


def test_전사_감정_접속_경로_선택도_지난_값으로_되살린다(시뮬):
    from app import App, 기억할_입력

    class 복원_dpg(가짜_dpg):
        def does_item_exist(self, tag):
            return True

        def get_item_configuration(self, tag):
            return {"items": {"asr": ["기본 전사", "맥 Qwen"], "emotion": ["기본 감정", "맥 em7"],
                              "link_route": ["같은 망", "NetBird"]}.get(tag)}

    assert {"asr", "emotion", "link_route"} <= set(기억할_입력)
    앱 = App(시뮬)
    앱.real = True  # 시뮬을 다시 띄우지 않고 값만 본다
    앱._영상들 = []
    가짜 = 복원_dpg()
    가짜.값.update({"real_volume": 50, **{f"speed_{키}": 0.0 for 키 in 앱._수동_속도값}})
    앱._입력을_되살린다(가짜, {"asr": "맥 Qwen", "emotion": "맥 em7", "link_route": "NetBird"})
    assert (가짜.값["asr"], 가짜.값["emotion"], 앱._경로) == ("맥 Qwen", "맥 em7", "NetBird")
