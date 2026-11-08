"""조회 패널이 고르는 값의 시험.

창을 띄우지 아니한다. 무엇을 그리는지가 아니라 무엇을 고르는지만 본다.
"""

import numpy as np
import pytest

from panels import 계층_이름, sections
from app import 새_런타임
from runtime import Command, ImuReading, Sensors, ServoReading, load_home_pose


def 센서(t=0.0) -> Sensors:
    영 = np.zeros(14)
    return Sensors(
        t=t,
        servo=ServoReading(
            angle=np.linspace(-0.5, 0.5, 14),
            velocity=영.copy(),
            load=영.copy(),
            voltage=np.full(14, 7.1),
            temperature=np.full(14, 30.0),
        ),
        imu=ImuReading(gyro=np.zeros(3), accel=np.array([0.0, 0.0, 9.81])),
        contact=np.array([True, True]),
    )


def test_한_주기도_안_돌았으면_그릴_것이_없다():
    assert sections(새_런타임().snapshot(), load_home_pose()[0]) == ()


def test_서보_표가_열네_줄이고_목표각을_같이_낸다():
    이름, _ = load_home_pose()
    런타임 = 새_런타임()
    런타임.step(센서())
    덩어리 = sections(런타임.snapshot(), 이름)[0]

    assert 덩어리.title == "서보"
    assert len(덩어리.rows) == 14
    assert 덩어리.rows[0][0] == 이름[0]
    assert "실제 (rad)" in 덩어리.header and "목표 (rad)" in 덩어리.header


def test_가속도_크기를_같이_낸다():
    """크기를 안 보면 IMU 가 죽어도 아무 신호가 나지 아니한다."""
    이름, _ = load_home_pose()
    런타임 = 새_런타임()
    런타임.step(센서())
    imu = sections(런타임.snapshot(), 이름)[1]
    assert dict(imu.rows)["가속도 크기"] == "9.810"


def test_모든_계층의_칸이_찬다():
    이름, _ = load_home_pose()
    런타임 = 새_런타임()
    런타임.step(센서())
    계층 = {d.title: d for d in sections(런타임.snapshot(), 이름)[2:]}
    런타임.close()

    assert list(계층) == [이름2 for _, 이름2 in 계층_이름]
    assert dict(계층["안전"].rows)["정지 여부"] == "아니오"
    assert dict(계층["지각"].rows)["사람"] == "없다"
    assert dict(계층["종합 판단"].rows)["모드"] == "IDLE"
    assert "따라가기 거리 도달" in dict(계층["전진 판단"].rows)
    assert dict(계층["걷기"].rows)["걷기 명령"].startswith("+0.00 +0.00 +0.00")


def test_계층이_서면_그_칸이_찬다():
    """계층이 자기 조회 자료형을 넣으면 패널이 고쳐지지 않고 따라간다."""
    from dataclasses import dataclass

    @dataclass
    class 안전조회:
        state: str
        voltage_min: float

    이름, _ = load_home_pose()
    런타임 = 새_런타임()
    런타임.step(센서())
    바뀐 = 런타임.snapshot().__class__(
        **{**vars(런타임.snapshot()), "safety": 안전조회(state="정상", voltage_min=7.1)}
    )
    안전 = sections(바뀐, 이름)[2]
    assert dict(안전.rows) == {"state": "정상", "voltage_min": "7.10"}


def test_표시용_박스는_표에_적지_않고_화면_좌표로_편다():
    from dataclasses import dataclass

    from panels import box_rects

    @dataclass
    class 지각조회:
        tracks: str
        boxes: tuple

    이름, _ = load_home_pose()
    런타임 = 새_런타임()
    런타임.step(센서())
    바뀐 = 런타임.snapshot().__class__(
        **{**vars(런타임.snapshot()),
           "perception": 지각조회(tracks="#1", boxes=((1, 0.25, 0.5, 0.5, 1.0),))}
    )
    런타임.close()

    지각 = sections(바뀐, 이름)[2 + [n for _, n in 계층_이름].index("지각")]
    assert dict(지각.rows) == {"사람": "#1"}
    assert box_rects(바뀐, 960, 480) == ((1, 240, 240, 480, 480),)


def test_조회가_없으면_그릴_박스도_없다():
    from panels import box_rects

    assert box_rects(None, 960, 480) == ()


def test_고른_대상의_번호를_조회에서_읽는다():
    from dataclasses import dataclass

    from panels import target_id

    @dataclass
    class 판단조회:
        target: str

    @dataclass
    class 조회:
        judgment: object

    assert target_id(조회(판단조회("#7 cx+0.10 cy-0.20 h0.40"))) == 7
    assert target_id(조회(판단조회("없다"))) is None
    assert target_id(None) is None


# --- 최근 기록 (사양 4-5) ---

from types import SimpleNamespace

from panels import History


def _기록용_조회(t, mode="IDLE", emotion_t=None, idle="없음", times=None, 글="", sound=None):
    return SimpleNamespace(
        t=t,
        judgment=SimpleNamespace(mode=mode, emotion_t=emotion_t, emotion="긍정 0.91 응답", idle_motion=idle),
        perception=SimpleNamespace(transcript_times=times, transcript=글),
        sound=sound,
    )


def test_최근_기록은_바뀐_것만_최근_것부터_적는다():
    기록 = History()
    기록.update(_기록용_조회(1.0, times=(0.1, 0.5, 0.9), 글="옛말"))  # 기준. 적지 않는다
    assert list(기록.lines) == []
    기록.update(_기록용_조회(2.0, mode="SEARCH_HEAD", times=(0.1, 0.5, 0.9), 글="옛말"))
    기록.update(_기록용_조회(3.0, mode="FOLLOW", times=(2.5, 2.9, 3.0), 글="안녕"))
    기록.update(_기록용_조회(4.0, mode="FOLLOW", emotion_t=3.9, idle="react_positive",
                       times=(2.5, 2.9, 3.0), 글="안녕", sound=(3.95, "애교")))
    기록.update(_기록용_조회(4.1, mode="FOLLOW", emotion_t=3.9, idle="react_positive",
                       times=(2.5, 2.9, 3.0), 글="안녕", sound=(3.95, "애교")))
    assert list(기록.lines) == [
        "    4.0  효과음 애교",
        "    4.0  표현 react_positive",
        "    4.0  감정 긍정 0.91 응답",
        "    3.0  전사 「안녕」",
        "    3.0  모드 SEARCH_HEAD → FOLLOW",
        "    2.0  모드 IDLE → SEARCH_HEAD",
    ]


def test_요약은_운영에_볼_값만_한_줄씩_낸다():
    from panels import summary_rows

    런타임 = 새_런타임()
    런타임.step(센서())
    줄 = dict(summary_rows(런타임.snapshot()))
    assert set(줄) == {"모드", "대상", "안전", "전사문", "감정", "표현", "서보 전압 최저", "서보 온도 최고"}
    assert 줄["안전"] in ("정상",) or 줄["안전"].startswith("정지")
    assert summary_rows(None) == ()
