"""월드와 온보드 SW 사이의 경계.

한 방향으로 센서 묶음을 만들고, 반대 방향으로 지령을 월드에 넣는다.
계약은 `AIRookie_Runtime/docs/spec/02_driver_contract.md` 가 정본이다.

**온보드 SW 는 이 파일을 알지 못하고, 이 파일은 온보드 SW 의 속을 알지
못한다.** 여기가 경계다. 실물에서는 진입점이 같은 자리에 선다.
"""

from __future__ import annotations

import mujoco
import numpy as np

from gait import CONTROL_DT
from runtime import Command, ImuReading, ManualInput, Sensors, ServoReading
from world import World


class Bridge:
    """월드를 센서 묶음으로 읽고, 지령을 월드에 적용한다.

    전압은 서보마다 하중으로 계산한다(사양 2-4). 온도는 물리 엔진에 없는 값이라
    기본값을 싣는다. 주입 패널이 넣은 값은 둘 다 계산값보다 앞선다.
    """

    # 서보 전류가 0 일 때 서보가 보고하는 전압이다. 실물 v2_15 걷기 로그의 서보 전압을
    # 포트 전류 합으로 회귀한 절편이 7.24 ~ 7.37 V 다 (Runtime `research/10` 9장).
    기본_전압_V = 7.3
    # 서보 전류 = 전류_계수 × max(|바깥 하중| − 전류_문턱, 0). 바깥 하중은 서보 힘에서
    # 관절 감쇠 × 속도와 armature × 가속을 뺀 값이다 (Playground `research/04` 20-12).
    전류_계수_A_per_Nm = 1.29
    전류_문턱_Nm = 0.417
    # 서보 전압 = 기본 − 포트 A 전류 합 × A 계수 − 포트 B 전류 합 × B 계수 (Ω).
    # 포트 B 는 오른다리 hip_roll · hip_pitch · knee · ankle, 나머지가 포트 A 다
    # (Runtime `research/10` 13장, 세 기록 평균).
    경로_저항_Ω = {
        "left_hip_yaw": (0.32, 0.03), "left_hip_roll": (0.38, 0.03), "left_hip_pitch": (0.47, 0.03),
        "left_knee": (0.54, 0.04), "left_ankle": (0.52, 0.03),
        "neck_pitch": (0.28, 0.04), "head_pitch": (0.19, 0.04), "head_yaw": (0.08, 0.02), "head_roll": (0.13, 0.03),
        "right_hip_yaw": (0.23, 0.02), "right_hip_roll": (0.03, 0.08), "right_hip_pitch": (0.03, 0.13),
        "right_knee": (0.02, 0.21), "right_ankle": (0.03, 0.20),
    }
    포트_B = ("right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
    # 실측이 없다. 잠정값이다.
    기본_온도_C = 30.0

    def __init__(self, world: World) -> None:
        self.world = world
        모델 = world.model

        # 액추에이터가 무는 관절의 주소를 한 번만 잡는다. 순서는 액추에이터
        # 순서이며 그것이 곧 홈 자세 파일의 차례다.
        self._축수 = int(모델.nu)
        관절_ids = [int(모델.actuator_trnid[i, 0]) for i in range(self._축수)]
        if any(i < 0 for i in 관절_ids):
            raise KeyError("액추에이터가 무는 관절을 찾을 수 없다")
        self._qpos_addr = np.array([모델.jnt_qposadr[i] for i in 관절_ids])
        self._qvel_addr = np.array([모델.jnt_dofadr[i] for i in 관절_ids])
        # 서보 엔코더는 출력축 각도를 읽는다. 학습 장면은 구동 관절 뒤에 유격 관절
        # (`<이름>_backlash`)을 두므로 그 각을 더한 값이 엔코더 값이다. 학습 관측도
        # 같은 합이다 (Playground `joystick.py` `_get_obs`). 유격 관절이 없는 축은 -1.
        유격_ids = [
            mujoco.mj_name2id(모델, mujoco.mjtObj.mjOBJ_JOINT, f"{모델.joint(i).name}_backlash")
            for i in 관절_ids
        ]
        self._유격_qpos_addr = np.array([모델.jnt_qposadr[i] if i >= 0 else -1 for i in 유격_ids])
        self._유격_있음 = self._유격_qpos_addr >= 0

        # 서보 속도 상한(실물 46번 레지스터, step/s). 0 이면 상한이 없다. 실물 `--servo-speed` 흉내다.
        self.servo_speed = 0
        # 서보 구동. 끄면 액추에이터 힘이 0 이다 (`set_armed`). 되돌릴 원래 이득을 쥔다.
        self.armed = True
        self._힘_원래 = (모델.actuator_gainprm[:, 0].copy(), 모델.actuator_biasprm[:, 1:3].copy())

        self._gyro_addr = world.sensor_address("gyro")
        self._accel_addr = world.sensor_address("accelerometer")

        # 부하의 분모다. 0 인 축은 나누지 못하므로 1 로 둔다.
        힘_상한 = np.abs(np.asarray(모델.actuator_forcerange[:, 1], dtype=np.float64))
        self._힘_상한 = np.where(힘_상한 > 0.0, 힘_상한, 1.0)

        # 토크 상한을 걸 때 곱할 기준이다. 모델의 값을 직접 줄이므로
        # 기준을 따로 들고 있어야 되돌릴 수 있다.
        self._기준_힘범위 = np.array(모델.actuator_forcerange, dtype=np.float64)
        self._토크_상한 = 1.0

        # 전압 계산 재료. 축 순서는 액추에이터 순서다.
        이름들 = [모델.actuator(i).name for i in range(self._축수)]
        self._감쇠 = np.array(모델.dof_damping[self._qvel_addr], dtype=np.float64)
        self._회전자_관성 = np.array(모델.dof_armature[self._qvel_addr], dtype=np.float64)
        self._포트_B = np.array([n in self.포트_B for n in 이름들])
        self._저항 = np.array([self.경로_저항_Ω[n] for n in 이름들], dtype=np.float64)

        # 주입한 전압. None 이면 계산값을 싣는다.
        self._전압: float | None = None
        self._온도 = float(self.기본_온도_C)

        # 표현 사양 2장, 표현 구현 계획 4-6: 지령의 ear·eye 를 들고 있는다.
        self.ear: tuple[float | None, float | None] = (None, None)
        self.eye: tuple[float, float] = (0.0, 0.0)

    # --- 앱이 주입으로 덮어쓰는 값 ---

    def set_voltage(self, volt: float | None) -> None:
        """서보가 보고할 전압을 정한다. 주입 패널이 부른다. None 이면 계산값으로 돌아간다."""
        self._전압 = None if volt is None else float(volt)

    def set_temperature(self, celsius: float) -> None:
        """서보가 보고할 온도를 정한다. 주입 패널이 부른다."""
        self._온도 = float(celsius)

    # --- 월드 → 온보드 SW ---

    def sensors(self, t: float, manual: ManualInput | None = None) -> Sensors:
        """한 주기 분량을 묶어 낸다.

        카메라와 마이크는 싣지 아니한다. 앱이 지각 입구에 따로 놓는다 (계약 2-3).
        """
        자료 = self.world.data

        힘 = np.asarray(자료.actuator_force, dtype=np.float64)
        부하 = np.clip(힘 / self._힘_상한, -1.0, 1.0)

        return Sensors(
            t=t,
            servo=ServoReading(
                angle=self._엔코더_각(자료.qpos),
                velocity=자료.qvel[self._qvel_addr],
                load=부하,
                voltage=self._서보_전압(자료) if self._전압 is None else np.full(self._축수, self._전압),
                temperature=np.full(self._축수, self._온도),
            ),
            imu=ImuReading(
                gyro=자료.sensordata[self._gyro_addr : self._gyro_addr + 3],
                accel=자료.sensordata[self._accel_addr : self._accel_addr + 3],
            ),
            # 발마다 하나씩 두 칸이다 (계약 2-5). 실물 스위치도 발마다 하나다.
            contact=np.array(self.world.foot_contacts()),
            manual=manual,
        )

    def _서보_전압(self, 자료) -> np.ndarray:
        바깥_하중 = (
            np.asarray(자료.actuator_force, dtype=np.float64)
            - self._감쇠 * 자료.qvel[self._qvel_addr]
            - self._회전자_관성 * 자료.qacc[self._qvel_addr]
        )
        전류 = self.전류_계수_A_per_Nm * np.maximum(np.abs(바깥_하중) - self.전류_문턱_Nm, 0.0)
        포트_전류 = np.array([전류[~self._포트_B].sum(), 전류[self._포트_B].sum()])
        return self.기본_전압_V - self._저항 @ 포트_전류

    def _엔코더_각(self, qpos: np.ndarray) -> np.ndarray:
        각 = np.array(qpos[self._qpos_addr], dtype=np.float64)
        각[self._유격_있음] += qpos[self._유격_qpos_addr[self._유격_있음]]
        return 각

    # --- 온보드 SW → 월드 ---

    def set_armed(self, on: bool) -> None:
        """서보 구동을 켜고 끈다. 끄면 액추에이터 힘이 0 이라 축 늘어진다 -- 실물 `--arm` 없이 띄운 서보 흉내다."""
        이득, 편향 = self._힘_원래
        모델 = self.world.model
        모델.actuator_gainprm[:, 0] = 이득 if on else 0.0
        모델.actuator_biasprm[:, 1:3] = 편향 if on else 0.0
        self.armed = bool(on)

    def apply(self, command: Command) -> None:
        """지령을 월드에 넣는다. 서보 속도 상한이 있으면 한 주기에 그 속도만큼만 목표를 옮긴다."""
        목표 = np.asarray(command.angle, dtype=np.float64)
        if self.servo_speed > 0:
            # 4096 step 이 한 바퀴다. 실물 서보는 목표를 받아도 이 속도 넘게 돌지 않는다.
            한_주기 = self.servo_speed * 2 * np.pi / 4096 * CONTROL_DT
            지금 = self.world.data.ctrl
            목표 = 지금 + np.clip(목표 - 지금, -한_주기, 한_주기)
        self.world.data.ctrl[:] = 목표

        # 표현 사양 2장, 표현 구현 계획 4-6: 귀와 눈 지령을 기억한다.
        self.ear = command.ear
        self.eye = command.eye

        # 같은 값이 연달아 와도 값싸게 넘어가야 한다 (계약 3장).
        if command.torque_limit != self._토크_상한:
            self.world.model.actuator_forcerange[:] = (
                self._기준_힘범위 * command.torque_limit
            )
            self._토크_상한 = command.torque_limit
