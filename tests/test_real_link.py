"""실물 연결 (real_link.py) 시험. 젯슨 대신 가짜 송출을 띄운다."""

from __future__ import annotations

import json
import re
import pickle
import socket
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from debug_link import recv_msg, send_msg
from real_link import Real
from runtime import Command, ImuReading, ServoReading, Snapshot


def _조회(각: np.ndarray) -> Snapshot:
    축 = np.zeros(14)
    return Snapshot(
        t=3.0,
        servo=ServoReading(angle=각, velocity=축, load=축, voltage=축 + 7.4, temperature=축 + 30, age=축),
        imu=ImuReading(gyro=np.zeros(3), accel=np.array([0.0, 0.0, 9.8])),
        command=Command(angle=각, ear=(10.0, -10.0), eye=(1.0, 0.5)),
        perception=SimpleNamespace(tracks="#7", boxes=((7, 0.1, 0.2, 0.3, 0.4),)),
    )


def _기다린다(조건, timeout: float = 10.0) -> bool:
    끝 = time.monotonic() + timeout
    while time.monotonic() < 끝:
        if 조건():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def 송출():
    """가짜 젯슨 송출. 붙으면 H · S · F 를 하나씩 보내고, 앱이 보낸 종류를 `받은` 에 모은다."""
    서버 = socket.create_server(("127.0.0.1", 0))
    port = 서버.getsockname()[1]
    각 = np.linspace(-0.3, 0.3, 14)
    그림 = np.zeros((480, 640, 3), np.uint8)
    그림[:, :, 0] = 200  # R
    ok, jpg = cv2.imencode(".jpg", 그림)
    assert ok
    받은: list[bytes] = []

    def 보낸다():
        conn, _ = 서버.accept()
        with conn:
            send_msg(conn, b"H", json.dumps({"policy": "v2_48_300M.onnx"}).encode())
            send_msg(conn, b"S", pickle.dumps((_조회(각), {"t": 3.0, "k": 150, "late": 2, "cycle_ms": 15.5})))
            send_msg(conn, b"F", struct.pack("<d", 3.0) + jpg.tobytes())
            # 앱이 닫을 때까지 쥐며 앱이 보낸 종류를 모은다.
            try:
                while True:
                    받은.append(recv_msg(conn)[0])
            except (ConnectionError, OSError):
                pass

    실 = threading.Thread(target=보낸다, daemon=True)
    실.start()
    yield port, 각, 받은
    서버.close()


def test_실물_조회와_프레임을_받아_자세와_패널을_채운다(송출):
    port, _, 받은 = 송출
    실물 = Real(host=None, port=port)
    try:
        assert _기다린다(lambda: 실물.due(0.0) == 1)
        실물.cycle(1)
        assert (실물.t, 실물.cycles, 실물.missed, 실물.cycle_ms) == (3.0, 150, 2, 15.5)
        assert 실물.policy.name == "v2_48_300M.onnx"
        assert 실물.state == "받는 중"
        assert 실물.status()[0].title == "서보"
        assert 실물.runtime.snapshot().perception.boxes[0][0] == 7

        assert _기다린다(lambda: 실물.frame is not None)
        그림 = 실물.frame
        assert 그림.shape == (480, 640, 3)
        assert abs(int(그림[..., 0].mean()) - 200) <= 3  # RGB 차례가 그대로 돌아온다
        assert 그림[..., 2].mean() < 5
        # 프레임을 받았다고 알린다. 젯슨은 이것이 와야 다음 프레임을 보낸다.
        assert _기다린다(lambda: 받은 == [b"A"])
    finally:
        실물.close()


def test_송출이_없으면_대기로_알린다():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    실물 = Real(host=None, port=port)
    try:
        assert _기다린다(lambda: 실물.state == "대기")
        assert 실물.due(0.0) == 0 and 실물.frame is None and 실물.status() == ()
    finally:
        실물.close()


def test_실행_명령은_이미_돌면_띄우지_않고_무늬는_자기를_잡지_않는다():
    from launcher import run_command, stop_command

    명령 = run_command("v2_48_300M.onnx", arm=True, safety=False)
    assert "--policy assets/v2_48_300M.onnx" in 명령 and "--arm" in 명령 and "--safety" not in 명령
    assert "--no-servo" not in 명령 and "--no-servo" in run_command("v2_48_300M.onnx", False, False, no_servo=True)
    assert "--servo-speed" not in 명령 and "--servo-speed 300" in run_command("v2_48_300M.onnx", True, False, servo_speed=300)
    assert "pgrep -f '[p]ython -m real[.]run'" in 명령 and "이미 실행 중" in 명령
    assert "setsid nohup" in 명령 and "PYTHONUNBUFFERED=1" in 명령
    # 이 명령을 도는 셸이 pgrep 에 걸려 '이미 실행 중' 이 되지 않게 한다.
    assert re.search(r"python -m real\.run", 명령) is None
    # 뒤에 띄운 프로세스는 SIGINT 를 무시한다. Ctrl-C 와 같은 길인 SIGTERM 으로 멈춘다.
    assert "pkill -TERM -f '[p]ython -m real[.]run'" in stop_command()


def test_실물_모드의_한_바퀴는_젯슨_시계와_실행기_상태를_그린다(송출):
    from test_app import 가짜_dpg

    from app import App

    class 대시보드_dpg(가짜_dpg):
        def get_item_configuration(self, tag):
            return {"items": self.값.get(f"items:{tag}", [])}

        def configure_item(self, tag, **kwargs):
            if "items" in kwargs:
                self.값[f"items:{tag}"] = kwargs["items"]
            super().configure_item(tag, **kwargs)

        def get_value(self, tag):
            return self.값.get(tag, "")

        def set_y_scroll(self, tag, value):
            pass

        def get_y_scroll_max(self, tag):
            return 0.0

    실행기 = SimpleNamespace(policies=["v2_48_300M.onnx", "v2_53s1_300M.onnx"], running=True, connected=True, address="jetson",
                          note="실행함", lines=["구동 시작. Ctrl-C 로 끝낸다"], close=lambda: None)
    port, _, _ = 송출
    앱 = App(Real(host=None, port=port), launcher=실행기)
    try:
        assert 앱.real and 앱.sim is 앱._실물  # noqa: SLF001
        assert _기다린다(lambda: 앱.sim.due(0.0) == 1)
        가짜 = 대시보드_dpg()
        앱._한_바퀴(가짜)  # noqa: SLF001
        assert "시각    3.00 s   주기 150   늦은 주기 2" in 가짜.값["clock"]
        assert 가짜.값["run_state"].startswith("실행 중")
        assert 가짜.값["run_policy"] == "v2_48_300M.onnx"  # 도는 정책을 고른 채로 둔다
        assert 가짜.값["run_log"] == "구동 시작. Ctrl-C 로 끝낸다"
        assert "spectator_tex" not in 가짜.값  # 실물에는 외부 시점이 없다
        assert 가짜.값["link_status"] == "젯슨: 연결됨 · jetson\n화면 전송: 받는 중\n실행 정책: v2_48_300M.onnx"
    finally:
        앱.sim.close()


def test_디버깅_명령과_주입_소리를_송출로_보낸다():
    """실물 디버깅 도구 (사양 4-5)."""
    from debug_link import recv_msg

    서버 = socket.create_server(("127.0.0.1", 0))
    port = 서버.getsockname()[1]
    받은 = []

    def 받는다():
        conn, _ = 서버.accept()
        with conn:
            send_msg(conn, b"H", json.dumps({"policy": "v2_48_300M.onnx"}).encode())
            while len(받은) < 2:
                받은.append(recv_msg(conn))

    실 = threading.Thread(target=받는다, daemon=True)
    실.start()
    실물 = Real(host=None, port=port)
    try:
        assert not 실물.send_command({"kind": "expression", "name": "heard"}) or 실물.state == "받는 중"
        assert _기다린다(lambda: 실물.state == "받는 중")
        assert 실물.send_command({"kind": "expression", "name": "heard"})
        assert 실물.send_mic(np.array([1, -2, 3], dtype=np.int16))
        assert _기다린다(lambda: len(받은) == 2)
        assert 받은[0] == (b"C", json.dumps({"kind": "expression", "name": "heard"}, ensure_ascii=False).encode())
        assert 받은[1][0] == b"M"
        np.testing.assert_array_equal(np.frombuffer(받은[1][1], "<i2"), [1, -2, 3])
    finally:
        실물.close()
        서버.close()


def test_송출에_붙지_않았으면_보내지_못한다():
    실물 = Real(host=None, port=1)
    try:
        assert not 실물.send_command({"kind": "sound", "name": "지침"})
    finally:
        실물.close()


def test_수동이면_키를_보내고_자율이면_보내지_아니한다():
    """사양 4-4. 수동 · 자율은 젯슨 조회를 따른다."""
    from debug_link import recv_msg

    서버 = socket.create_server(("127.0.0.1", 0))
    port = 서버.getsockname()[1]
    받은 = []

    def 받는다():
        conn, _ = 서버.accept()
        with conn:
            send_msg(conn, b"H", json.dumps({"policy": "p.onnx"}).encode())
            while len(받은) < 2:
                받은.append(json.loads(recv_msg(conn)[1]))

    threading.Thread(target=받는다, daemon=True).start()
    실물 = Real(host=None, port=port)
    try:
        assert _기다린다(lambda: 실물.state == "받는 중")
        실물.drive(forward=1.0, lateral=0.0, turn=0.0)       # 자율 — 안 간다
        assert 실물.set_manual(True)
        실물.manual = True                                   # 조회가 돌아온 셈
        실물.drive(forward=1.0, lateral=0.0, turn=0.0)
        실물.drive(forward=1.0, lateral=0.0, turn=0.0)       # 그대로면 간격 안에 다시 안 간다
        assert _기다린다(lambda: len(받은) == 2)
        assert 받은[0] == {"kind": "manual", "on": True}
        assert 받은[1]["kind"] == "keys" and 받은[1]["forward"] == 1.0 and 받은[1]["ear"] == [False, False]
    finally:
        실물.close()
        서버.close()
