"""오프스크린 뷰와 궤도 카메라.

로봇 머리 카메라(head_cam) 시점과 구경꾼 궤도 카메라 시점을
MuJoCo 오프스크린 렌더러로 렌더하고 numpy 배열로 돌려준다.
GUI 라이브러리(dearpygui)를 일절 쓰지 아니한다.
"""

from __future__ import annotations

import math
from typing import Sequence

import mujoco
import numpy as np

# --- 잠정 상수 ---
# 눈으로 보고 고칠 값이다.
_ORBIT_DRAG_SENSITIVITY = 0.12  # 잠정: 마우스 드래그 1픽셀당 방위각·고도 변화량 (도/px)
_ORBIT_ZOOM_RATIO = 0.95        # 잠정: 휠 한 칸(1 tick)당 거리 배율 (< 1 이면 확대)
_ORBIT_PAN_SENSITIVITY = 0.002  # 잠정: 주시점 이동 1픽셀당 평면 이동 거리 (m/px)
_ORBIT_MIN_DISTANCE = 0.3       # 궤도 카메라 거리 하한 (m, 0 이하 뒤집힘 방지)
_PERSON_HEIGHT = 1.7            # 사람 키 (m, spec 및 person.obj 정본)
_PERSON_WIDTH = 0.5             # 잠정: 사람 어깨/가로 폭 (m)

# --- 귀와 눈 (표현 사양 1장) ---
# 모델의 실제 부품을 쓴다. 귀는 CAD 귀 + 서보 암 메시(ear_L·ear_R)를 서보 혼 축(ear_*_axis site)
# 둘레로 돌리고, 눈은 LED 가 비치는 원판(eye_L·eye_R)을 밝기대로 칠한다. 메시와 축은
# AIRookie_CAD/scripts/apply_meshes.py 가 CAD 에서 넣는다.
_EYE_DARK = np.array([0.25, 0.25, 0.27], dtype=np.float32)  # 꺼진 눈 (모델의 'other' 색)
_EYE_LIT = np.array([1.0, 0.95, 0.8], dtype=np.float32)     # 켜진 눈


def _require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str, 무엇: str) -> int:
    """이름으로 번호를 찾되 못 찾으면 즉시 예외를 던진다.

    mj_name2id 는 실패하면 -1 을 주고 음수 인덱싱이 조용히 마지막 요소를 읽으므로,
    못 찾은 경우 즉시 KeyError 로 바꾼다.
    """
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise KeyError(f"{무엇}를 찾을 수 없다: {name!r}")
    return idx


class OrbitCamera:
    """구경꾼 시점의 궤도 카메라. 방위각·고도·거리·주시점을 든다.

    기본 시점은 **로봇 뒤에서 로봇과 사람을 사선으로 본다.**

    방위각 0 도가 로봇 정뒤다. 정뒤에서는 사람이 로봇에 가려 겹치므로 30 도
    틀었다. 사람이 화면에 들어오는 구간은 300 ~ 60 도뿐이고, 그 밖에서는
    사람이 뒤로 빠진다. 주시점을 로봇보다 조금 앞에 두어 둘이 같이 들어온다.
    """

    def __init__(
        self,
        azimuth: float = 330.0,
        elevation: float = -15.0,
        distance: float = 3.0,
        lookat: tuple[float, float, float] | Sequence[float] = (0.5, 0.0, 0.4),
    ) -> None:
        self.azimuth = float(azimuth) % 360.0
        self.elevation = float(np.clip(elevation, -89.0, 89.0))
        self.distance = max(_ORBIT_MIN_DISTANCE, float(distance))
        self.lookat = np.array(lookat, dtype=np.float64)

    def drag(self, dx: float, dy: float) -> None:
        """마우스 이동 픽셀 -> 방위각·고도.

        고도는 -89 ~ +89 도 안에 가두고, 방위각은 360 도로 감는다.
        """
        self.azimuth = float((self.azimuth + dx * _ORBIT_DRAG_SENSITIVITY) % 360.0)
        self.elevation = float(
            np.clip(self.elevation + dy * _ORBIT_DRAG_SENSITIVITY, -89.0, 89.0)
        )

    def pan(self, dx: float, dy: float) -> None:
        """주시점을 화면 평면으로 민다."""
        rad_az = math.radians(self.azimuth)
        rad_el = math.radians(self.elevation)

        # 화면 오른쪽 벡터 (월드 좌표계)
        right = np.array([math.sin(rad_az), -math.cos(rad_az), 0.0], dtype=np.float64)

        # 화면 위쪽 벡터 (월드 좌표계, right x forward)
        up = np.array(
            [
                -math.sin(rad_el) * math.cos(rad_az),
                -math.sin(rad_el) * math.sin(rad_az),
                math.cos(rad_el),
            ],
            dtype=np.float64,
        )

        delta = (-dx * right + dy * up) * _ORBIT_PAN_SENSITIVITY
        self.lookat = self.lookat + delta

    def zoom(self, ticks: float) -> None:
        """휠 -> 거리. 하한 0.3 m 아래로 내려가지 아니한다."""
        self.distance = max(
            _ORBIT_MIN_DISTANCE,
            float(self.distance * (_ORBIT_ZOOM_RATIO ** float(ticks))),
        )

    def to_mjv(self) -> mujoco.MjvCamera:
        """MuJoCo 렌더러에 넘길 MjvCamera 구조체를 만든다."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = float(self.azimuth)
        cam.elevation = float(self.elevation)
        cam.distance = float(self.distance)
        cam.lookat[:] = self.lookat
        return cam


class Views:
    """두 시점의 오프스크린 렌더러와 사람 박스 계산을 다룬다.

    **로봇 시점은 실물 카메라와 같은 4:3 이다** (계약 6-1). 모델의 `head_cam` 은
    세로 화각만 정하므로(51.1°) 가로 화각은 비율이 정하고, 4:3 에서 64.9° 로
    실물과 같다. 해상도만 실물(3264×2464)보다 낮다. 구경꾼 시점은 2:1 이다.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_size: tuple[int, int] = (640, 480),
        spectator_size: tuple[int, int] = (960, 480),
        ear_rest_deg: float = 0.0,
    ) -> None:
        self.model = model
        self.robot_size = (int(robot_size[0]), int(robot_size[1]))
        self.spectator_size = (int(spectator_size[0]), int(spectator_size[1]))
        self.camera = OrbitCamera()

        self._head_cam_id = _require_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam", "카메라")
        self._person_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, "person", "몸체")
        self._person_mocap_id = int(model.body_mocapid[self._person_body_id])
        if self._person_mocap_id < 0:
            raise KeyError("사람이 mocap 몸체가 아니다")

        self._ear_geoms = tuple(
            _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, n, "귀 부품") for n in ("ear_L", "ear_R")
        )
        self._ear_axes = tuple(
            _require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{n}_axis", "귀 축") for n in ("ear_L", "ear_R")
        )
        self._eye_geoms = tuple(
            _require_id(model, mujoco.mjtObj.mjOBJ_GEOM, n, "눈 부품") for n in ("eye_L", "eye_R")
        )
        # 힘을 놓은 귀는 마지막 자리에 있다. 처음에는 쉬는 각도다 (표현 사양 4-2).
        self._ear_last_deg = [float(ear_rest_deg), float(ear_rest_deg)]

        self._robot_renderer = mujoco.Renderer(
            model, height=self.robot_size[1], width=self.robot_size[0]
        )
        self._spectator_renderer = mujoco.Renderer(
            model, height=self.spectator_size[1], width=self.spectator_size[0]
        )

        fovy_rad = math.radians(float(model.cam_fovy[self._head_cam_id]))
        self._tan_half_fovy = math.tan(fovy_rad / 2.0)
        self._tan_half_fovx = self._tan_half_fovy * (self.robot_size[0] / self.robot_size[1])

    def robot_frame(
        self,
        data: mujoco.MjData,
        ear: tuple[float | None, float | None] = (None, None),
        eye: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """모델의 head_cam 시점 프레임을 낸다. HxWx3 uint8 배열이다.

        귀를 앞으로 크게 돌리면 화면 가장자리에 들 수 있어 여기서도 귀와 눈을 입힌다.
        """
        self._robot_renderer.update_scene(data, camera=self._head_cam_id)
        self._귀와_눈을_입힌다(self._robot_renderer.scene, data, ear, eye)
        return self._robot_renderer.render()

    def _귀와_눈을_입힌다(
        self,
        scene: mujoco.MjvScene,
        data: mujoco.MjData,
        ear: tuple[float | None, float | None],
        eye: tuple[float, float],
    ) -> None:
        """장면의 귀 부품을 명령 각도로 돌리고 눈 부품을 밝기대로 칠한다 (표현 사양 1장).

        귀 각도는 도이고 0 이 CAD 기준 자세(위로 선 귀), + 가 귀 끝이 앞이다. 좌우 같은
        부호다. None(힘 놓음)이면 마지막 자리에 둔다.
        """
        for i in range(2):
            if ear[i] is not None:
                self._ear_last_deg[i] = float(ear[i])
        회전 = {}
        for i, (g, s) in enumerate(zip(self._ear_geoms, self._ear_axes)):
            축 = np.asarray(data.site_xmat[s], dtype=np.float64).reshape(3, 3)[:, 2]
            # 축은 바깥 방향이다. 왼쪽 귀(+왼쪽 축)는 + 회전이 앞, 오른쪽 귀는 − 회전이 앞이다.
            각 = math.radians(self._ear_last_deg[i]) * (1.0 if i == 0 else -1.0)
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, 축, 각)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            회전[g] = (R.reshape(3, 3), np.asarray(data.site_xpos[s], dtype=np.float64))
        밝기 = {g: float(min(1.0, max(0.0, eye[i]))) for i, g in enumerate(self._eye_geoms)}

        for k in range(scene.ngeom):
            geom = scene.geoms[k]
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                continue
            if geom.objid in 회전:
                R, p = 회전[geom.objid]
                geom.pos[:] = p + R @ (np.asarray(geom.pos, dtype=np.float64) - p)
                geom.mat[:] = R @ np.asarray(geom.mat, dtype=np.float64).reshape(3, 3)
            elif geom.objid in 밝기:
                b = 밝기[geom.objid]
                geom.rgba[:3] = _EYE_DARK + (_EYE_LIT - _EYE_DARK) * b
                geom.emission = b

    def spectator_frame(
        self,
        data: mujoco.MjData,
        ear: tuple[float | None, float | None] = (None, None),
        eye: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """구경꾼 궤도 카메라 시점 프레임을 낸다. HxWx3 uint8 배열이다."""
        mjv_cam = self.camera.to_mjv()
        mujoco.mjv_updateCamera(self.model, data, mjv_cam, self._spectator_renderer.scene)
        self._spectator_renderer.update_scene(data, camera=mjv_cam)
        self._귀와_눈을_입힌다(self._spectator_renderer.scene, data, ear, eye)
        return self._spectator_renderer.render()

    def person_box(self, data: mujoco.MjData) -> tuple[float, float, float, float] | None:
        """사람이 로봇 카메라 화면 어디에 잡히는지 정규화 좌표 (cx, cy, w, h) 로 낸다.

        가운데가 0, 오른쪽이 +, 아래가 + 다.
        화면 밖이거나 카메라 뒤면 None 을 낸다.
        """
        person_pos = np.asarray(data.mocap_pos[self._person_mocap_id], dtype=np.float64)

        cam_pos = np.asarray(data.cam_xpos[self._head_cam_id], dtype=np.float64)
        cam_mat = np.asarray(data.cam_xmat[self._head_cam_id], dtype=np.float64).reshape(3, 3)

        offset = person_pos - cam_pos
        # cam_mat 열:
        # col 0: 오른쪽 (+X)
        # col 1: 위쪽 (+Y)
        # col 2: 뒤쪽 (+Z) -> 전방(forward)은 -col 2
        x_cam = float(cam_mat[:, 0] @ offset)
        y_cam = float(cam_mat[:, 1] @ offset)
        z_forward = float(-cam_mat[:, 2] @ offset)

        if z_forward <= 0.0:
            return None

        cx = (x_cam / z_forward) / self._tan_half_fovx
        cy = (-y_cam / z_forward) / self._tan_half_fovy

        screen_height_m = 2.0 * z_forward * self._tan_half_fovy
        screen_width_m = 2.0 * z_forward * self._tan_half_fovx
        h = min(1.0, float(_PERSON_HEIGHT) / screen_height_m)
        w = min(1.0, float(_PERSON_WIDTH) / screen_width_m)

        # 중심이 아니라 박스로 본다. 중심만 보면 몸이 절반 걸쳐 있는데도
        # 없는 것으로 판정하며, 검출 모델은 잘린 사람도 잡는다.
        if abs(cx) - w > 1.0 or abs(cy) - h > 1.0:
            return None

        return (float(cx), float(cy), float(w), float(h))

    def close(self) -> None:
        """렌더러 자원을 해제한다."""
        self._robot_renderer.close()
        self._spectator_renderer.close()
