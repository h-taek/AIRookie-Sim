"""외부 주입 트리거 시험.

검증 모드 가르기, 프레임·소리 흘려 넣기, 전압 강하, 온도 상승을
본다. 사양은 `docs/spec/01_simulator.md` 3장이다.

입력원 자체(프레임 시퀀스, wav, 마이크, 카메라)는 `test_sources.py` 가 본다.
여기서는 주입이 경계를 넘는 것만 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from bridge import Bridge
from inject import Injector, 모드_감정, 모드_추종
from sources import VideoFile, WavFile
from world import World


def 가짜_시퀀스(개수: int = 4, duration_s: float = 1.0) -> VideoFile:
    프레임들 = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(개수)]
    return VideoFile(프레임들, fps=개수 / duration_s, loop=False)


# --- 검증 모드 ---


def test_기본은_추종_모드다():
    assert Injector(bridge=None).mode == 모드_추종


def test_추종_모드에서는_프레임을_내지_아니한다():
    """그 모드의 프레임은 월드 렌더다. 주입이 낼 것이 아니다."""
    injector = Injector(bridge=None)
    injector.play(t=0.0, frames=가짜_시퀀스())

    assert injector.mode == 모드_추종
    assert injector.frame(0.0) is None


def test_감정_모드에서_프레임이_나온다():
    injector = Injector(bridge=None)
    injector.set_mode(모드_감정)
    injector.play(t=0.0, frames=가짜_시퀀스(개수=4, duration_s=1.0))

    첫_장 = injector.frame(0.0)
    assert 첫_장 is not None
    assert 첫_장[0, 0, 0] == 0


def test_모르는_모드는_막는다():
    with pytest.raises(ValueError):
        Injector(bridge=None).set_mode("아무거나")


def test_재생_시작_시각부터_센다():
    """시뮬이 10 초 돈 뒤에 틀어도 발화는 그 발화의 처음부터 나간다."""
    injector = Injector(bridge=None)
    injector.set_mode(모드_감정)
    injector.play(t=10.0, frames=가짜_시퀀스(개수=4, duration_s=1.0))

    첫_장 = injector.frame(10.0)
    assert 첫_장 is not None
    assert 첫_장[0, 0, 0] == 0

    # 0.25 초 뒤가 둘째 칸이다.
    둘째 = injector.frame(10.3)
    assert 둘째 is not None
    assert 둘째[0, 0, 0] == 1


# --- 소리 ---


def test_소리는_모드를_가리지_아니한다():
    """추종 모드에서도 호출어를 넣어야 한다. 듣고 돌아보는 것이 추종의 시작이다."""
    injector = Injector(bridge=None)
    injector.play(t=0.0, audio=WavFile(np.ones(1600, dtype=np.float32), 16000))

    assert injector.mode == 모드_추종
    블록 = injector.audio_block(0.02)
    assert 블록 is not None
    assert len(블록) == 960


def test_표본율이_달라도_마이크_형으로_한_주기_분량을_낸다():
    """44.1 kHz 소리도 마이크 자리에는 int16 48 kHz 로 20 ms 분량이 실린다 (계약 2-3)."""
    injector = Injector(bridge=None)
    injector.play(t=0.0, audio=WavFile(np.full(44100, 0.5, dtype=np.float32), 44100))

    블록 = injector.audio_block(0.02)
    assert 블록 is not None
    assert len(블록) == 960 and 블록.dtype == np.int16
    assert 블록[0] == round(0.5 * 32767)


def test_소리를_안_넣었으면_None이다():
    assert Injector(bridge=None).audio_block(0.02) is None


def test_새_소리를_넣으면_앞의_것이_닫힌다():
    injector = Injector(bridge=None)
    앞 = WavFile(np.ones(1600, dtype=np.float32), 16000)
    injector.play(t=0.0, audio=앞)
    injector.audio_block(0.02)

    뒤 = WavFile(np.full(1600, -0.5, dtype=np.float32), 16000)
    injector.play(t=0.0, audio=뒤)

    블록 = injector.audio_block(0.02)
    assert 블록 is not None
    np.testing.assert_array_equal(블록, np.full(960, round(-0.5 * 32767), dtype=np.int16))


def test_멈추면_아무것도_안_나온다():
    injector = Injector(bridge=None)
    injector.set_mode(모드_감정)
    injector.play(
        t=0.0,
        frames=가짜_시퀀스(),
        audio=WavFile(np.ones(1600, dtype=np.float32), 16000),
    )
    assert injector.playing is True

    injector.stop()
    assert injector.playing is False
    assert injector.frame(0.0) is None
    assert injector.audio_block(0.02) is None


def test_멈추면_입력원을_닫는다():
    """카메라와 마이크는 열어 둔 채 두면 장치를 붙잡고 있는다."""

    class 닫히는지_보는_것:
        sample_rate = 16000

        def __init__(self):
            self.닫혔다 = False

        def block(self, n):
            return None

        def close(self):
            self.닫혔다 = True

    소리 = 닫히는지_보는_것()
    injector = Injector(bridge=None)
    injector.play(t=0.0, audio=소리)
    injector.stop()

    assert 소리.닫혔다 is True


# --- 전압 · 온도 ---


def test_전압_온도를_주입하면_bridge가_내는_센서_묶음에_그_값이_실린다():
    world = World()
    bridge = Bridge(world)
    injector = Injector(bridge)

    injector.drop_voltage(6.2)
    injector.raise_temperature(58.0)

    sensors = bridge.sensors(t=0.0)
    assert np.all(sensors.servo.voltage == pytest.approx(6.2))
    assert np.all(sensors.servo.temperature == pytest.approx(58.0))


def test_clear하면_계산_전압과_기본_온도로_돌아온다():
    world = World()
    bridge = Bridge(world)
    injector = Injector(bridge)
    계산값 = bridge.sensors(t=0.0).servo.voltage

    injector.drop_voltage(5.5)
    injector.raise_temperature(65.0)
    injector.clear()

    sensors = bridge.sensors(t=0.0)
    np.testing.assert_allclose(sensors.servo.voltage, 계산값)
    assert np.all(sensors.servo.temperature == pytest.approx(Bridge.기본_온도_C))

