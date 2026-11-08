"""시뮬레이터 물리 월드.

MuJoCo 모델을 올리고 물리를 미는 일만 한다.
센서 묶음을 만들거나 지령을 받는 일은 bridge.py 의 몫이다.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
# 학습 장면이다(`runner.py` 의 기본 task). 비-backlash 판은 서보 kp 가 13.37 로
# 학습 조건(17.11)이 아니다 -- 같은 정책이 더 물렁하게 움직인다.
_DEFAULT_SCENE = (
    _REPO_ROOT.parent
    / "AIRookie_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"
)
_DEFAULT_PERSON_MESH = _REPO_ROOT / "assets/person.obj"

# 사람 메시는 키 1.7 m 에 원점이 몸의 중심이다 (tools/prepare_person_mesh.py).
# 바닥(z=0) 위에 세우면 몸 중심 높이는 그 절반인 0.85 m 다.
_PERSON_CENTER_Z = 0.85

# 오프스크린 버퍼의 상한이다. 장면이 정하지 않으면 MuJoCo 기본값 640×480 이라
# 그보다 큰 시점을 렌더할 수 없다. **렌더 크기는 앱이 정할 일이고** 장면은
# 학습 저장소의 것이라 고칠 자리가 아니므로, 올릴 때 여기서 연다.
_오프스크린_상한 = (1920, 1080)
_DEFAULT_PERSON_POS = (5.0, 0.0)

# 메시는 항등 자세에서 +x 를 바라본다. 사람이 로봇 앞(+x)에 서 있으면 등을
# 보이게 되므로, 기본은 로봇 쪽을 바라보도록 돌려 놓는다.
_MESH_FORWARD_YAW = 0.0


class World:
    """MuJoCo 물리 월드를 관리한다.

    장면 XML에 사람 mocap 몸체를 덧붙여 올리고, 물리를 전진시키며,
    관절과 사람 위치를 조회·조작하는 인터페이스를 제공한다.
    """

    def __init__(
        self,
        scene_path: str | Path | None = None,
        person_mesh: str | Path | None = None,
    ) -> None:
        resolved_scene = (
            Path(scene_path).resolve()
            if scene_path is not None
            else _DEFAULT_SCENE.resolve()
        )
        if not resolved_scene.exists():
            raise FileNotFoundError(f"장면 파일을 찾을 수 없다: {resolved_scene}")

        resolved_mesh = (
            Path(person_mesh).resolve()
            if person_mesh is not None
            else _DEFAULT_PERSON_MESH.resolve()
        )
        if not resolved_mesh.exists():
            raise FileNotFoundError(
                f"사람 메시 파일이 없다: {resolved_mesh}. "
                "tools/prepare_person_mesh.py 로 만들어야 한다"
            )

        spec = mujoco.MjSpec.from_file(str(resolved_scene))
        spec.meshdir = str((resolved_scene.parent / "assets").resolve())
        spec.visual.global_.offwidth = _오프스크린_상한[0]
        spec.visual.global_.offheight = _오프스크린_상한[1]

        mesh = spec.add_mesh()
        mesh.name = "person_mesh"
        mesh.file = str(resolved_mesh)

        body = spec.worldbody.add_body(
            name="person",
            mocap=True,
            pos=[_DEFAULT_PERSON_POS[0], _DEFAULT_PERSON_POS[1], _PERSON_CENTER_Z],
        )
        # 사람은 구경거리이며 로봇과 충돌하지 않는다.
        body.add_geom(
            name="person_body",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="person_mesh",
            rgba=[0.80, 0.66, 0.55, 1.0],
            contype=0,
            conaffinity=0,
        )

        self.model: mujoco.MjModel = spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # 초기 자세는 장면의 home 키프레임을 쓴다. 없으면 로봇이 엉뚱한
        # 자세로 서고 그 위에서 잰 값이 전부 무의미해지므로 즉시 터뜨린다.
        self._home_key = self._require_id(mujoco.mjtObj.mjOBJ_KEY, "home", "키프레임")

        self._actuator_names: tuple[str, ...] = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        )

        self._person_mocap_id = int(
            self.model.body_mocapid[self._require_id(mujoco.mjtObj.mjOBJ_BODY, "person", "몸체")]
        )
        self._torso_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "base", "몸체")
        self._발_ids = tuple(
            self._require_id(mujoco.mjtObj.mjOBJ_BODY, 이름, "몸체")
            for 이름 in ("foot_assembly", "foot_assembly_2")
        )
        self._바닥_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "floor", "몸체")

        self.reset()

    def reset(self) -> None:
        """로봇을 home 키프레임에, 사람을 기본 자리에 둔다. 시각도 0 이다."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        # 키프레임이 사람 자리를 담는지에 기대지 아니한다. 사람은 장면을
        # 올린 뒤에 덧붙인 몸체라 키프레임이 모르는 것이다.
        self.data.mocap_pos[self._person_mocap_id] = (
            _DEFAULT_PERSON_POS[0], _DEFAULT_PERSON_POS[1], _PERSON_CENTER_Z
        )
        mujoco.mj_forward(self.model, self.data)
        # 기본은 로봇을 바라본다.
        self.set_person_yaw(self.로봇을_바라보는_각())

    def hold_home(self) -> None:
        """로봇을 home 키프레임 자세에 묶어 둔다. 사람(mocap)은 그대로다. 실물 `--no-servo` 흉내다."""
        self.data.qpos[:] = self.model.key_qpos[self._home_key]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _require_id(self, kind, name: str, 무엇: str) -> int:
        """이름으로 번호를 찾되 못 찾으면 즉시 터뜨린다.

        mj_name2id 는 실패하면 -1 을 주고 음수 인덱싱이 조용히 마지막 요소를
        읽으므로, 못 찾은 것이 엉뚱한 값으로 바뀌어 시험까지 통과한다.
        """
        idx = mujoco.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise KeyError(f"{무엇}를 찾을 수 없다: {name!r}")
        return idx

    def sensor_address(self, name: str) -> int:
        """이름에 해당하는 센서가 `sensordata` 의 몇 번째 자리에서 시작하는지 낸다."""
        return int(self.model.sensor_adr[
            self._require_id(mujoco.mjtObj.mjOBJ_SENSOR, name, "센서")
        ])

    @property
    def actuator_names(self) -> tuple[str, ...]:
        """액추에이터 순서대로의 이름 목록이다."""
        return self._actuator_names

    def actuator_index(self, name: str) -> int:
        """이름에 해당하는 액추에이터 번호를 돌려준다.

        mj_name2id 가 못 찾으면 -1 을 주므로, 못 찾으면 즉시 KeyError 를 던진다.
        """
        return self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name, "액추에이터")

    def step(self, dt: float) -> None:
        """dt(초) 만큼 물리를 전진시킨다."""
        if dt <= 0:
            raise ValueError(f"dt 는 양수여야 한다: {dt}")
        steps = max(1, round(dt / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def foot_contacts(self) -> tuple[bool, bool]:
        """왼발 · 오른발이 바닥에 닿았는가.

        뷰어의 `get_feet_contacts` 와 같은 판정이다 -- 발 몸체의 도형과 바닥
        몸체의 도형 사이에 접촉이 하나라도 있는가. 정책이 이 값으로 학습했다.
        """
        몸체 = self.model.geom_bodyid
        닿음 = [False, False]
        for i in range(self.data.ncon):
            접촉 = self.data.contact[i]
            짝 = {int(몸체[접촉.geom1]), int(몸체[접촉.geom2])}
            if self._바닥_id not in 짝:
                continue
            for 발, 발_id in enumerate(self._발_ids):
                if 발_id in 짝:
                    닿음[발] = True
        return 닿음[0], 닿음[1]

    def torso_height(self) -> float:
        """몸통의 현재 z 높이(m)를 낸다."""
        return float(self.data.xpos[self._torso_body_id, 2])

    @property
    def person_pos(self) -> tuple[float, float]:
        """사람 mocap 몸체의 현재 (x, y) 자리다."""
        return (
            float(self.data.mocap_pos[self._person_mocap_id, 0]),
            float(self.data.mocap_pos[self._person_mocap_id, 1]),
        )

    def set_person_pos(self, x: float, y: float) -> None:
        """사람 mocap 몸체의 (x, y) 자리를 옮긴다. 방향은 그대로 둔다."""
        self.data.mocap_pos[self._person_mocap_id, 0] = float(x)
        self.data.mocap_pos[self._person_mocap_id, 1] = float(y)
        mujoco.mj_forward(self.model, self.data)

    @property
    def person_yaw(self) -> float:
        """사람이 바라보는 방향이다. z 축 회전, rad."""
        w, _, _, z = self.data.mocap_quat[self._person_mocap_id]
        return float(2.0 * math.atan2(z, w)) - _MESH_FORWARD_YAW

    def set_person_yaw(self, yaw: float) -> None:
        """사람이 바라보는 방향을 정한다. z 축 회전, rad."""
        각 = float(yaw) + _MESH_FORWARD_YAW
        self.data.mocap_quat[self._person_mocap_id] = (
            math.cos(각 / 2.0), 0.0, 0.0, math.sin(각 / 2.0)
        )
        mujoco.mj_forward(self.model, self.data)

    def 로봇을_바라보는_각(self) -> float:
        """사람이 로봇 쪽을 보는 z 축 회전을 낸다."""
        x, y = self.person_pos
        로봇_x, 로봇_y = float(self.data.xpos[self._torso_body_id, 0]), float(
            self.data.xpos[self._torso_body_id, 1]
        )
        return math.atan2(로봇_y - y, 로봇_x - x)
