"""경계 시험.

월드에서 센서 묶음을 만들고 지령을 월드에 돌려주는 계약을 확인한다.
계약은 AIRookie_Runtime 의 docs/spec/02_driver_contract.md 다.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from bridge import Bridge
from app import 새_런타임
from runtime import Command, load_home_pose
from world import World



@pytest.fixture
def 다리():
    return Bridge(World())


def test_묶음의_모양이_계약과_맞다(다리):
    묶음 = 다리.sensors(t=0.0)
    for 칸 in (
        묶음.servo.angle,
        묶음.servo.velocity,
        묶음.servo.load,
        묶음.servo.voltage,
        묶음.servo.temperature,
    ):
        assert 칸.shape == (14,)
    assert 묶음.imu.gyro.shape == (3,)
    assert 묶음.imu.accel.shape == (3,)
    assert 묶음.t == 0.0


def test_관절각이_홈_자세에서_시작한다(다리):
    _, 홈 = load_home_pose()
    np.testing.assert_allclose(다리.sensors(t=0.0).servo.angle, 홈, atol=1e-6)


def test_관절각의_자리가_액추에이터_순서와_같다(다리):
    """한 축만 움직여 그 자리의 값만 바뀌는지 본다.

    주소를 잘못 잡으면 모양은 맞는데 자리가 밀린다. 모양만 보는 시험은
    그것을 잡지 못한다.
    """
    이름, _ = load_home_pose()
    자리 = 이름.index("left_knee")
    앞 = 다리.sensors(t=0.0).servo.angle.copy()

    다리.world.data.qpos[다리.world.model.jnt_qposadr[
        다리.world.model.actuator_trnid[자리, 0]
    ]] += 0.1
    mujoco.mj_forward(다리.world.model, 다리.world.data)

    뒤 = 다리.sensors(t=0.0).servo.angle
    바뀐_자리 = np.flatnonzero(np.abs(뒤 - 앞) > 1e-9)
    assert list(바뀐_자리) == [자리]


def test_서_있기만_해도_무릎에_부하가_걸린다(다리):
    """부하는 실측이 있는 값이다 — 홈 자세 유지만으로 무릎이 0.270 이었다.

    액추에이터 힘을 힘 상한으로 나눈 값이라 눈금은 실물과 다르지만,
    0 이 아니어야 하고 [-1, 1] 안이어야 한다.
    """
    이름, _ = load_home_pose()
    다리.apply(새_런타임().step(다리.sensors(t=0.0)))
    for _ in range(50):
        다리.world.step(0.02)

    부하 = 다리.sensors(t=1.0).servo.load
    assert np.all(np.abs(부하) <= 1.0)
    assert abs(부하[이름.index("left_knee")]) > 0.0


def test_온도는_물리에_없어_기본값이_실린다(다리):
    묶음 = 다리.sensors(t=0.0)
    assert np.all(묶음.servo.temperature == pytest.approx(Bridge.기본_온도_C))


def test_전압은_하중으로_처지고_경로가_긴_왼무릎이_더_처진다(다리):
    """서 있기만 해도 무릎이 몸무게를 받아 전류를 쓴다 (사양 2-4)."""
    이름, _ = load_home_pose()
    다리.apply(새_런타임().step(다리.sensors(t=0.0)))
    for _ in range(50):
        다리.world.step(0.02)

    전압 = 다리.sensors(t=1.0).servo.voltage
    왼, 오 = 전압[이름.index("left_knee")], 전압[이름.index("right_knee")]
    assert 왼 < 오 < Bridge.기본_전압_V
    assert np.all(전압 > 6.0)


def test_전류가_없으면_기본_전압이다(다리):
    다리.world.data.actuator_force[:] = 0.0
    다리.world.data.qvel[:] = 0.0
    다리.world.data.qacc[:] = 0.0
    assert np.all(다리.sensors(t=0.0).servo.voltage == pytest.approx(Bridge.기본_전압_V))


def test_주입한_전압과_온도가_묶음에_실린다(다리):
    다리.set_voltage(6.2)
    다리.set_temperature(71.0)
    묶음 = 다리.sensors(t=0.0)
    assert np.all(묶음.servo.voltage == pytest.approx(6.2))
    assert np.all(묶음.servo.temperature == pytest.approx(71.0))


def test_카메라와_마이크는_묶음에_싣지_않는다(다리):
    """지각 입구에 따로 놓는다 (계약 2-3)."""
    묶음 = 다리.sensors(t=0.0)
    assert not hasattr(묶음, "frame") and not hasattr(묶음, "audio")


def test_토크_상한이_힘_상한에_걸린다(다리):
    기준 = 다리.world.model.actuator_forcerange.copy()
    _, 홈 = load_home_pose()

    다리.apply(Command(angle=홈, torque_limit=0.5))

    np.testing.assert_allclose(다리.world.model.actuator_forcerange, 기준 * 0.5)

    다리.apply(Command(angle=홈, torque_limit=1.0))
    np.testing.assert_allclose(다리.world.model.actuator_forcerange, 기준)


def test_껍데기를_물려_백_주기를_돌려도_서_있다(다리):
    껍데기 = 새_런타임()
    for k in range(100):
        다리.apply(껍데기.step(다리.sensors(t=k * 0.02)))
        다리.world.step(0.02)
    assert 다리.world.torso_height() > 0.10


def test_발_접촉이_두_칸으로_실린다(다리):
    """발마다 하나다 (계약 2-5). 실물 스위치도 발마다 하나다."""
    묶음 = 다리.sensors(t=0.0)
    assert 묶음.contact.shape == (2,)
    assert 묶음.contact.dtype == bool
    assert tuple(묶음.contact) == 다리.world.foot_contacts()


def test_지령의_귀와_눈을_들고_있는다(다리):
    다리.apply(Command(angle=np.zeros(다리.world.model.nu), ear=(0.5, None), eye=(1.0, 0.2)))
    assert 다리.ear == (0.5, None) and 다리.eye == (1.0, 0.2)


def test_처음에는_귀_힘_없음_눈_꺼짐이다(다리):
    assert 다리.ear == (None, None) and 다리.eye == (0.0, 0.0)



def test_서보_속도_상한은_한_주기에_움직일_목표를_자른다(다리):
    """실물 46번 레지스터 흉내. step/s 를 4096 step/바퀴로 rad/s 로 바꿔 한 주기 몫만 움직인다."""
    from gait import CONTROL_DT

    _, 홈 = load_home_pose()
    다리.servo_speed = 300
    한_주기 = 300 * 2 * np.pi / 4096 * CONTROL_DT
    다리.apply(Command(angle=홈 + 1.0))
    assert np.allclose(다리.world.data.ctrl, 홈 + 한_주기)
    다리.apply(Command(angle=홈 - 1.0))
    assert np.allclose(다리.world.data.ctrl, 홈)


def test_서보_속도_상한이_0_이면_목표를_그대로_넣는다(다리):
    _, 홈 = load_home_pose()
    다리.apply(Command(angle=홈 + 1.0))
    assert np.allclose(다리.world.data.ctrl, 홈 + 1.0)


def test_서보_구동을_끄면_힘이_빠져_주저앉고_다시_켜면_돌아온다(다리):
    """실물 `--arm` 없음 흉내. 액추에이터 힘을 0 으로 둔다."""
    from gait import CONTROL_DT

    _, 홈 = load_home_pose()
    처음_높이 = 다리.world.torso_height()
    다리.set_armed(False)
    for _ in range(int(2.0 / CONTROL_DT)):
        다리.apply(Command(angle=홈))
        다리.world.step(CONTROL_DT)
    assert 다리.world.torso_height() < 처음_높이 - 0.03
    다리.set_armed(True)
    assert np.all(다리.world.model.actuator_gainprm[:, 0] > 0)


def test_서보_없이면_로봇을_홈_키프레임에_묶어_둔다(다리):
    """실물 `--no-servo` 흉내. 물리를 밀어도 홈 자세 그대로이고 센서도 홈 값이다."""
    from gait import CONTROL_DT

    _, 홈 = load_home_pose()
    for _ in range(int(1.0 / CONTROL_DT)):
        다리.apply(Command(angle=홈 + 0.3))
        다리.world.step(CONTROL_DT)
        다리.world.hold_home()
    assert np.allclose(다리.sensors(t=0.0).servo.angle, 홈, atol=1e-6)
