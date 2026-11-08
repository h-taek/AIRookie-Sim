"""물리 월드 시험.

MuJoCo 모델 적재, 액추에이터 순서 정합성, 물리 진행, 사람 mocap 조작을 검증한다.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from runtime import load_home_pose
from world import World


@pytest.fixture
def 월드():
    return World()


def test_액추에이터_순서가_파일_순서와_같다(월드):
    이름_목록, _ = load_home_pose()
    assert 월드.actuator_names == 이름_목록


def test_없는_액추에이터를_찾으면_즉시_터진다(월드):
    with pytest.raises(KeyError):
        월드.actuator_index("없는관절")


def test_백_주기를_돌려도_서_있다(월드):
    이름_목록, 각도_목록 = load_home_pose()
    for 이름, 각도 in zip(이름_목록, 각도_목록):
        인덱스 = 월드.actuator_index(이름)
        월드.data.ctrl[인덱스] = 각도

    for _ in range(100):
        월드.step(0.02)

    assert 월드.torso_height() > 0.10


def test_사람_자리를_옮길_수_있다(월드):
    월드.set_person_pos(3.0, -1.5)
    assert 월드.person_pos == pytest.approx((3.0, -1.5))


def test_홈에_서면_두_발이_다_닿아_있다(월드):
    assert 월드.foot_contacts() == (True, True)


def test_몸을_들어_올리면_두_발이_다_떨어진다(월드):
    """접촉이 늘 참이면 정책이 스윙 구간을 모른다."""
    월드.data.qpos[2] += 0.2
    mujoco.mj_forward(월드.model, 월드.data)
    assert 월드.foot_contacts() == (False, False)


def test_학습_장면을_띄운다(월드):
    """비-backlash 판은 kp 13.37 이라 학습 조건(17.11)이 아니다."""
    import numpy as np

    kp = 월드.model.actuator_gainprm[:, 0]
    np.testing.assert_allclose(kp, 17.11)
