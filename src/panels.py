"""좌·우 패널.

좌측은 월드 조작과 외부 주입, 우측은 로봇 상태 조회다.

**값을 고르는 일과 화면에 그리는 일을 가른다.** 앞엣것은 순수 함수라 창
없이 시험할 수 있고, 뒤엣것만 dearpygui 를 만진다. 조회는 읽기 전용이며
아무것도 바꾸지 아니한다 (계약 5장).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# 계층이 서기 전에 그 칸에 적을 말이다.
아직_없다 = "아직 없다"

# 조회에 실리지만 표에 적지 않는 칸이다. 로봇 카메라 화면에 그린다.
그리는_칸 = frozenset({"boxes"})

# 사양 2-3 의 다섯 줄이다. 칸 이름은 `Snapshot` 의 것이다.
계층_이름 = (
    ("safety", "안전"),
    ("perception", "지각"),
    ("judgment", "종합 판단"),
    ("approach", "전진 판단"),
    ("gait", "걷기"),
)

# 계층마다 화면에 적을 칸 이름과 차례다. 칸 이름은 계층 조회 자료형의 것이다.
# 여기 없는 칸은 원래 이름 그대로 뒤에 붙인다.
칸_이름 = {
    "safety": {
        "stopped": "정지 여부",
        "reason": "정지 까닭",
        "tilt_deg": "기울기 (°)",
        "imu_age_s": "IMU 나이 (s)",
        "calm_s": "안정 유지 (s)",
    },
    "perception": {
        "tracks": "사람",
        "tracks_age_s": "사람 나이 (s)",
        "faces": "얼굴 수",
        "speaking": "말하는 중",
        "transcript": "들은 말",
        "transcript_age_s": "들은 말 나이 (s)",
        "error": "오류",
    },
    "judgment": {
        "mode": "모드",
        "target": "대상",
        "heard": "들은 명령",
        "lost_s": "놓친 시간 (s)",
        "emotion": "감정",
        "emotion_age_s": "감정 나이 (s)",
        "emotion_note": "감정 메모",
        "idle_motion": "대기 동작",
    },
    "approach": {
        "command": "이동 명령",
        "at_follow_distance": "따라가기 거리 도달",
        "distance_m": "사람 거리 (m)",
        "body_turn_deg": "몸통 회전 (°)",
        "expression": "표현",
        "ear_eye": "귀·눈",
    },
    "gait": {
        "command": "걷기 명령",
        "phase_step": "걸음 위상",
    },
}


@dataclass(frozen=True)
class Section:
    """패널 한 덩어리. 머리줄이 있으면 표, 없으면 이름과 값의 나열이다."""

    title: str
    header: tuple[str, ...] | None
    rows: tuple[tuple[str, ...], ...]


def servo_section(snapshot, joint_names: tuple[str, ...]) -> Section:
    """서보 14축의 다섯 칸을 표로 낸다. 목표각을 같이 놓아 따라가는지 본다."""
    서보 = snapshot.servo
    목표 = snapshot.command.angle
    줄 = tuple(
        (
            이름,
            f"{서보.angle[i]:+.3f}",
            f"{목표[i]:+.3f}",
            f"{서보.velocity[i]:+.2f}",
            f"{서보.load[i]:+.2f}",
            f"{서보.voltage[i]:.2f}",
            f"{서보.temperature[i]:.0f}",
        )
        for i, 이름 in enumerate(joint_names)
    )
    return Section(
        title="서보",
        header=("관절", "실제 (rad)", "목표 (rad)", "속도 (rad/s)", "부하", "전압 (V)", "온도 (°C)"),
        rows=줄,
    )


def imu_section(snapshot) -> Section:
    """자이로와 가속도를 낸다. 크기를 같이 낸다 — IMU 가 죽으면 0 근처가 된다."""
    자이로 = snapshot.imu.gyro
    가속 = snapshot.imu.accel
    return Section(
        title="IMU",
        header=None,
        rows=(
            ("각속도 (rad/s)", " ".join(f"{v:+.3f}" for v in 자이로)),
            ("가속도 (m/s²)", " ".join(f"{v:+.3f}" for v in 가속)),
            ("가속도 크기", f"{float(np.linalg.norm(가속)):.3f}"),
        ),
    )


def layer_sections(snapshot) -> tuple[Section, ...]:
    """계층별 현재 상태다. 선 계층만 값이 있고 나머지는 비어 있다."""
    덩어리 = []
    for 칸, 이름 in 계층_이름:
        값 = getattr(snapshot, 칸)
        if 값 is None:
            줄 = ((아직_없다, ""),)
        else:
            이름표 = 칸_이름.get(칸, {})
            차례 = {k: i for i, k in enumerate(이름표)}
            항목 = sorted(
                ((k, v) for k, v in vars(값).items() if k not in 그리는_칸),
                key=lambda kv: (차례.get(kv[0], len(차례)), kv[0]),
            )
            줄 = tuple((이름표.get(k, k), _값_글(v)) for k, v in 항목)
        덩어리.append(Section(title=이름, header=None, rows=줄))
    return tuple(덩어리)


def _값_글(v) -> str:
    """조회 값을 칸에 적을 글로 바꾼다. 실수는 소수 둘째 자리까지다."""
    if v is None:
        return "없음"
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}"
    if isinstance(v, tuple) and v and all(isinstance(x, (float, np.floating)) for x in v):
        return " ".join(f"{x:+.2f}" for x in v)
    return str(v)


def box_rects(snapshot, width: int, height: int) -> tuple[tuple[int, int, int, int, int], ...]:
    """로봇 카메라 화면에 그릴 사람 박스다. (id, x1, y1, x2, y2) 픽셀.

    지각 조회의 표시용 박스를 화면 크기로 편다. 없으면 비어 있다.
    """
    지각 = None if snapshot is None else snapshot.perception
    if 지각 is None:
        return ()
    return tuple(
        (번호, int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
        for 번호, x1, y1, x2, y2 in getattr(지각, "boxes", ())
    )


def target_id(snapshot) -> int | None:
    """종합 판단이 고른 트랙 번호다. 조회 글에서 읽는다. 고른 것이 없으면 없다."""
    판단 = None if snapshot is None else snapshot.judgment
    글 = getattr(판단, "target", "") or ""
    if not 글.startswith("#"):
        return None
    try:
        return int(글[1:].split()[0])
    except ValueError:
        return None


def summary_rows(snapshot) -> tuple[tuple[str, str], ...]:
    """우측 패널 맨 위 요약이다. 운영 중에 볼 값만 한 줄씩 낸다. 스냅샷이 없으면 비어 있다.

    젯슨 쪽 판에 따라 칸이 없을 수 있다. 없는 칸은 '아직 없다' 로 둔다.
    """
    if snapshot is None:
        return ()

    def 칸(계층, 이름: str) -> str:
        값 = getattr(getattr(snapshot, 계층, None), 이름, None)
        return 아직_없다 if 값 is None else (str(값) or "—")

    안전 = getattr(snapshot, "safety", None)
    if 안전 is None:
        안전_글 = 아직_없다
    elif getattr(안전, "stopped", False):
        안전_글 = f"정지 · {안전.reason}" if getattr(안전, "reason", None) else "정지"
    else:
        안전_글 = "정상"
    return (
        ("모드", 칸("judgment", "mode")),
        ("대상", 칸("judgment", "target")),
        ("안전", 안전_글),
        ("전사문", 칸("perception", "transcript")),
        ("감정", 칸("judgment", "emotion")),
        ("표현", 칸("approach", "expression")),
        ("서보 전압 최저", f"{float(np.min(snapshot.servo.voltage)):.2f} V"),
        ("서보 온도 최고", f"{float(np.max(snapshot.servo.temperature)):.0f} °C"),
    )


def sections(snapshot, joint_names: tuple[str, ...]) -> tuple[Section, ...]:
    """우측 패널이 그릴 것 전부다. 스냅샷이 없으면 비어 있다."""
    if snapshot is None:
        return ()
    return (
        servo_section(snapshot, joint_names),
        imu_section(snapshot),
    ) + layer_sections(snapshot)


class History:
    """조회에서 바뀐 것을 젯슨 시각과 함께 모은다. 실물 디버깅 도구의 최근 기록이다 (사양 4-5).

    모드 전이, 전사문, 감정 판정, 표현, 효과음을 적는다. 처음 받은 조회는 기준으로만 쓴다 —
    붙기 전에 있던 일을 지금 일처럼 적지 아니한다. 최근 것이 앞이다.
    """

    def __init__(self, keep: int = 60) -> None:
        self.lines: deque[str] = deque(maxlen=keep)
        self._앞: dict | None = None

    def update(self, snapshot) -> None:
        if snapshot is None:
            return
        판단 = getattr(snapshot, "judgment", None)
        지각 = getattr(snapshot, "perception", None)
        지금 = {
            "mode": getattr(판단, "mode", None),
            "emotion_t": getattr(판단, "emotion_t", None),
            "idle_motion": getattr(판단, "idle_motion", None),
            "transcript_times": getattr(지각, "transcript_times", None),
            "sound": getattr(snapshot, "sound", None),
        }
        앞, self._앞 = self._앞, 지금
        if 앞 is None:
            return
        t = snapshot.t
        if 지금["mode"] != 앞["mode"] and 지금["mode"] is not None:
            self._적는다(t, f"모드 {앞['mode']} → {지금['mode']}")
        if 지금["transcript_times"] != 앞["transcript_times"] and 지금["transcript_times"] is not None:
            self._적는다(t, f"전사 「{getattr(지각, 'transcript', '')}」")
        if 지금["emotion_t"] != 앞["emotion_t"] and 지금["emotion_t"] is not None:
            self._적는다(t, f"감정 {판단.emotion}")
        if 지금["idle_motion"] != 앞["idle_motion"] and 지금["idle_motion"] not in (None, "없음"):
            self._적는다(t, f"표현 {지금['idle_motion']}")
        if 지금["sound"] != 앞["sound"] and 지금["sound"] is not None:
            self._적는다(지금["sound"][0], f"효과음 {지금['sound'][1]}")

    def _적는다(self, t: float, 글: str) -> None:
        self.lines.appendleft(f"{t:7.1f}  {글}")
