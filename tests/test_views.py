"""views.py 시험.

오프스크린 렌더링 결과의 형태와 화소 값, 사람 검출 박스(person_box) 투영,
궤도 카메라 조작(OrbitCamera)의 각도·거리 범위 제약을 검증한다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from views import (
    _ORBIT_DRAG_SENSITIVITY,
    _ORBIT_MIN_DISTANCE,
    OrbitCamera,
    Views,
)
from world import World


def test_두_시점_배열_크기와_타입():
    """로봇 시점과 구경꾼 시점 프레임이 (H, W, 3) 모양에 uint8 형인지 확인한다."""
    world = World()
    views = Views(world.model, robot_size=(640, 480), spectator_size=(320, 240))
    try:
        r_frame = views.robot_frame(world.data)
        s_frame = views.spectator_frame(world.data)

        assert r_frame.shape == (480, 640, 3)
        assert r_frame.dtype == np.uint8
        assert s_frame.shape == (240, 320, 3)
        assert s_frame.dtype == np.uint8
    finally:
        views.close()


def test_로봇_시점과_구경꾼_시점_픽셀_다름():
    """로봇 시점과 구경꾼 시점의 카메라 위치가 다르므로 렌더 결과 화소가 달라야 한다."""
    world = World()
    views = Views(world.model, robot_size=(640, 480), spectator_size=(640, 480))
    try:
        r_frame = views.robot_frame(world.data)
        s_frame = views.spectator_frame(world.data)

        assert not np.array_equal(r_frame, s_frame)
    finally:
        views.close()


def test_사람_위치_이동시_로봇_시점_프레임_변화():
    """사람을 로봇 앞으로 옮기면 로봇 머리 카메라 시점 프레임이 바뀐다."""
    world = World()
    views = Views(world.model)
    try:
        # 초기 사람 위치(5.0, 0.0)에서 프레임 확보
        frame_before = views.robot_frame(world.data).copy()

        # 사람을 로봇 정면 2.0 m 자리로 이동
        world.set_person_pos(2.0, 0.0)
        frame_after = views.robot_frame(world.data)

        assert not np.array_equal(frame_before, frame_after)
    finally:
        views.close()


def test_사람_앞뒤_위치에_따른_person_box():
    """사람을 로봇 앞에 두면 person_box 가 나오고, 뒤로 옮기면 None 이 나온다."""
    world = World()
    views = Views(world.model)
    try:
        # 로봇 앞
        world.set_person_pos(2.5, 0.0)
        box_front = views.person_box(world.data)
        assert box_front is not None
        cx, cy, w, h = box_front
        assert -1.0 <= cx <= 1.0
        assert -1.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0

        # 로봇 뒤 (카메라 뒤편)
        world.set_person_pos(-2.0, 0.0)
        box_back = views.person_box(world.data)
        assert box_back is None
    finally:
        views.close()


def test_사람_오른쪽_이동시_cx_증가():
    """사람을 로봇 시야 오른쪽으로 옮기면 person_box 의 cx 가 커진다."""
    world = World()
    views = Views(world.model)
    try:
        # 정면
        world.set_person_pos(2.5, 0.0)
        box_center = views.person_box(world.data)
        assert box_center is not None

        # 로봇 시야 기준 오른쪽은 월드 -y 방향
        world.set_person_pos(2.5, -0.4)
        box_right = views.person_box(world.data)
        assert box_right is not None

        # 로봇 시야 기준 왼쪽은 월드 +y 방향
        world.set_person_pos(2.5, 0.4)
        box_left = views.person_box(world.data)
        assert box_left is not None

        assert box_right[0] > box_center[0] > box_left[0]
    finally:
        views.close()


def test_사람_화면_밖_이동시_person_box_none():
    """사람이 시야각 밖으로 완전히 벗어나면 person_box 가 None 을 낸다."""
    world = World()
    views = Views(world.model)
    try:
        # 화각 밖으로 멀리 이동
        world.set_person_pos(2.5, 10.0)
        assert views.person_box(world.data) is None
    finally:
        views.close()


def test_궤도_카메라_드래그_방위각_고도_제한():
    """드래그가 방위각을 바꾸고 360도로 감으며, 고도는 ±89도 안에 가둔다."""
    cam = OrbitCamera(azimuth=160.0, elevation=-20.0)

    # 방위각 변화 확인
    cam.drag(dx=10.0, dy=0.0)
    assert not math.isclose(cam.azimuth, 160.0)

    # 방위각 360도 감기 확인
    cam.drag(dx=360.0 / _ORBIT_DRAG_SENSITIVITY, dy=0.0)
    assert 0.0 <= cam.azimuth < 360.0

    # 고도 상한 (+89도) 초과 드래그
    cam.drag(dx=0.0, dy=1000.0)
    assert math.isclose(cam.elevation, 89.0)

    # 고도 하한 (-89도) 미만 드래그
    cam.drag(dx=0.0, dy=-2000.0)
    assert math.isclose(cam.elevation, -89.0)


def test_궤도_카메라_줌_거리_변화_및_하한():
    """휠 조작이 거리를 바꾸고, 거리 하한(0.3 m) 밑으로 내려가지 아니한다."""
    cam = OrbitCamera(distance=2.5)

    # 줌 인 (거리 감소)
    cam.zoom(ticks=1.0)
    assert cam.distance < 2.5

    # 줌 아웃 (거리 증가)
    cam.zoom(ticks=-2.0)
    assert cam.distance > 2.5

    # 과도한 줌 인 시 하한 0.3 m 유지
    cam.zoom(ticks=100.0)
    assert math.isclose(cam.distance, _ORBIT_MIN_DISTANCE)


def test_궤도_카메라_팬_주시점_이동():
    """팬 조작이 주시점을 화면 평면 방향으로 민다."""
    cam = OrbitCamera(lookat=(0.0, 0.0, 0.3))
    initial_lookat = cam.lookat.copy()

    cam.pan(dx=50.0, dy=-30.0)
    assert not np.array_equal(cam.lookat, initial_lookat)


def test_궤도_카메라_to_mjv_변환():
    """MjvCamera 구조체로 올바르게 변환된다."""
    cam = OrbitCamera(azimuth=120.0, elevation=30.0, distance=3.0, lookat=(1.0, 2.0, 0.5))
    mjv = cam.to_mjv()

    assert mjv.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert math.isclose(mjv.azimuth, 120.0)
    assert math.isclose(mjv.elevation, 30.0)
    assert math.isclose(mjv.distance, 3.0)
    assert np.allclose(mjv.lookat, [1.0, 2.0, 0.5])


def test_필수_요소_부재시_keyerror():
    """모델에 필수 카메라나 몸체가 없으면 KeyError 를 던진다."""
    empty_model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(KeyError):
        Views(empty_model)


@pytest.fixture
def 월드():
    return World()


@pytest.fixture
def 시점들(월드):
    v = Views(월드.model)
    try:
        yield v
    finally:
        v.close()


def _장면_부품(시점들, 이름):
    """구경꾼 장면에서 모델 부품 하나의 그림 geom 을 찾는다."""
    import mujoco

    gid = mujoco.mj_name2id(시점들.model, mujoco.mjtObj.mjOBJ_GEOM, 이름)
    scene = 시점들._spectator_renderer.scene
    for k in range(scene.ngeom):
        g = scene.geoms[k]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == gid:
            return g
    raise AssertionError(f"장면에 {이름} 이 없다")


def test_귀_각도를_바꾸면_구경꾼_화면이_바뀐다(시점들, 월드):
    눕힌 = 시점들.spectator_frame(월드.data, ear=(-60.0, -60.0))
    세운 = 시점들.spectator_frame(월드.data, ear=(30.0, 30.0))
    assert np.any(눕힌 != 세운)


def test_귀는_서보_축을_중심으로_돌고_양수가_앞이다(시점들, 월드):
    m, d = 시점들.model, 월드.data
    앞 = d.cam_xmat[m.camera("head_cam").id].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
    for 이름 in ("ear_L", "ear_R"):
        축점 = d.site_xpos[m.site(f"{이름}_axis").id]
        시점들.spectator_frame(d, ear=(0.0, 0.0))
        g = _장면_부품(시점들, 이름)
        영_자리, 영_반경 = np.array(g.pos), np.linalg.norm(np.array(g.pos) - 축점)
        시점들.spectator_frame(d, ear=(60.0, 60.0))
        g = _장면_부품(시점들, 이름)
        assert np.linalg.norm(np.array(g.pos) - 축점) == pytest.approx(영_반경, abs=1e-6)
        assert (np.array(g.pos) - 영_자리) @ 앞 > 0.0, f"{이름}: + 가 앞이 아니다"


def test_귀가_없음이면_마지막_자리에_둔다(시점들, 월드):
    시점들.spectator_frame(월드.data, ear=(30.0, 30.0))
    든 = 시점들.spectator_frame(월드.data, ear=(30.0, 30.0))
    놓은 = 시점들.spectator_frame(월드.data, ear=(None, None))
    assert np.array_equal(든, 놓은)


def test_눈은_LED_원판을_밝기대로_칠한다(시점들, 월드):
    시점들.spectator_frame(월드.data, eye=(1.0, 0.0))
    왼, 오 = _장면_부품(시점들, "eye_L"), _장면_부품(시점들, "eye_R")
    assert 왼.emission == pytest.approx(1.0) and 오.emission == pytest.approx(0.0)
    assert 왼.rgba[0] > 오.rgba[0]
