"""걷기 계층을 월드에 물려 대본 명령으로 걷게 한다 (계획 5-1 통과 판정).

위 계층이 없어 온보드 SW 는 명령 0 을 낸다. 대본 명령은 여기서 걷기 계층에
직접 넣는다. 경계(`Bridge`)는 앱과 같은 것을 쓴다.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from app import 기본_정책, 시범_동작
from bridge import Bridge
from gait import CONTROL_DT, Gait
from runtime import Command, load_policy_pair
from world import World

# 30 초다. 넘어졌다고 보는 몸통 높이는 홈에서 선 높이(약 0.15 m)의 2/3 쯤이다.
구간_s = 30.0
넘어진_높이 = 0.10


def 걷게_한다(명령) -> tuple[World, float]:
    """명령을 물고 구간만큼 걷는다. 가장 낮았던 몸통 높이를 같이 낸다."""
    월드 = World()
    다리 = Bridge(월드)
    걷기 = Gait(기본_정책, 시범_동작, load_policy_pair()[1])
    가장_낮은 = 월드.torso_height()
    for k in range(int(구간_s / CONTROL_DT)):
        묶음 = 다리.sensors(t=k * CONTROL_DT)
        목표 = 걷기.step(
            t=묶음.t, angle=묶음.servo.angle, velocity=묶음.servo.velocity,
            gyro=묶음.imu.gyro, accel=묶음.imu.accel, contact=묶음.contact,
            command=np.asarray(명령, dtype=np.float64),
        )
        다리.apply(Command(angle=목표))
        월드.step(CONTROL_DT)
        가장_낮은 = min(가장_낮은, 월드.torso_height())
    return 월드, 가장_낮은


def 몸통(월드):
    return 월드.data.body("base")


@pytest.mark.parametrize(
    "이름, 명령",
    [
        ("정지", [0, 0, 0, 0, 0, 0, 0]),
        ("전진", [0.15, 0, 0, 0, 0, 0, 0]),
        ("좌회전", [0, 0, 0.5, 0, 0, 0, 0]),
        ("우회전", [0, 0, -0.5, 0, 0, 0, 0]),
        ("머리 요", [0, 0, 0, 0, 0, 0.2, 0]),
        ("머리 피치", [0, 0, 0, 0, 0.2, 0, 0]),
    ],
)
def test_대본_명령으로_30_초_넘어지지_아니한다(이름, 명령):
    _, 가장_낮은 = 걷게_한다(명령)
    assert 가장_낮은 > 넘어진_높이, f"{이름}: 몸통이 {가장_낮은:.3f} m 까지 내려갔다"


def test_전진_명령으로_앞으로_간다():
    월드, _ = 걷게_한다([0.15, 0, 0, 0, 0, 0, 0])
    x, y, _ = 몸통(월드).xpos
    assert x > 1.0, f"30 초에 {x:.2f} m 만 갔다"


def test_회전_명령은_양쪽_다_돈다():
    """한쪽만 재면 틀린다 (HANDOFF 재론 금지). 홈이 어긋난 정책은 한쪽이 안 돈다."""
    for 부호 in (+1, -1):
        월드 = World()
        다리 = Bridge(월드)
        걷기 = Gait(기본_정책, 시범_동작, load_policy_pair()[1])
        방향들 = []
        for k in range(int(10.0 / CONTROL_DT)):
            묶음 = 다리.sensors(t=k * CONTROL_DT)
            목표 = 걷기.step(
                t=묶음.t, angle=묶음.servo.angle, velocity=묶음.servo.velocity,
                gyro=묶음.imu.gyro, accel=묶음.imu.accel, contact=묶음.contact,
                command=np.array([0, 0, 0.5 * 부호, 0, 0, 0, 0], dtype=np.float64),
            )
            다리.apply(Command(angle=목표))
            월드.step(CONTROL_DT)
            R = 몸통(월드).xmat.reshape(3, 3)
            방향들.append(np.arctan2(R[1, 0], R[0, 0]))
        # 마지막 방향만 보면 ±180 도에서 접힌다. 펼쳐 누적한다 (research/04 3장).
        누적_도 = np.degrees(np.unwrap(방향들)[-1] - 방향들[0])
        assert 누적_도 * 부호 > 90.0, f"회전 {0.5 * 부호:+.1f} 에 {누적_도:+.0f} 도"
