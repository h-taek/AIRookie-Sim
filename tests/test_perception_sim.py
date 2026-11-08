"""시뮬 로봇 카메라 프레임으로 실제 검출기를 돌린다 (계획 5-2 통과 판정).

사람을 옮기면 트랙의 cx · h 가 따라 바뀌는지 본다. 검출 모델은 온보드 저장소의
`tools/export_detector.py` 로 만든다. 없으면 건너뛴다.
"""

import math
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from runtime import load_autonomy
from views import Views
from vision import Detector, Tracker
from world import World

_MODEL = Path(__file__).resolve().parents[2] / "AIRookie_Runtime" / "models" / "yolov8n.onnx"
pytestmark = pytest.mark.skipif(not _MODEL.exists(), reason="검출 모델이 없다")


@pytest.fixture(scope="module")
def 눈():
    월드 = World()
    시점 = Views(월드.model)
    설정 = load_autonomy()["perception"]
    검출 = Detector(_MODEL, conf=float(설정["detector_conf"]), nms_iou=float(설정["detector_nms_iou"]))
    yield 월드, 시점, 검출
    시점.close()


def 본다(눈, x, y):
    월드, 시점, 검출 = 눈
    월드.set_person_pos(x, y)
    월드.set_person_yaw(월드.로봇을_바라보는_각())
    프레임 = 시점.robot_frame(월드.data)
    높이, 너비 = 프레임.shape[:2]
    return Tracker(0.3).update(검출.detect(프레임), 너비, 높이, t=0.0)


def test_로봇_시점은_실물과_같은_4_대_3_이다(눈):
    _, 시점, _ = 눈
    assert 시점.robot_size == (640, 480)


def test_정면의_사람을_찾는다(눈):
    트랙 = 본다(눈, 5.0, 0.0)
    assert len(트랙) == 1
    assert abs(트랙[0].cx) < 0.1


def test_로봇의_오른쪽에_서면_cx_가_플러스다(눈):
    """로봇은 +x 를 본다. 오른쪽은 −y 다 (자율 사양 2-3: 오른쪽이 +)."""
    오른 = 본다(눈, 5.0, -1.2)
    왼 = 본다(눈, 5.0, 1.2)
    assert 오른[0].cx > 0.2 and 왼[0].cx < -0.2


def test_가까워지면_h_가_커진다(눈):
    먼 = 본다(눈, 7.0, 0.0)[0].h
    가까운 = 본다(눈, 4.0, 0.0)[0].h
    assert 가까운 > 먼


def test_h_는_세로_화각으로_셈한_값과_맞다(눈):
    """세로 화각 51.1° 에서 키 1.7 m 가 거리 d 에서 채우는 비율 = 1.7 / (2 d tan 25.55°) (계약 6-1)."""
    거리 = 5.0
    트랙 = 본다(눈, 거리, 0.0)[0]
    기대 = 1.7 / (2 * 거리 * math.tan(math.radians(51.1 / 2)))
    assert 트랙.h == pytest.approx(기대, rel=0.15)
