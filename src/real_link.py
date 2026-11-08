"""실물 연결 (사양 4-4).

젯슨에서 도는 실물 진입점의 송출(Runtime `debug_link.py`)에 붙어, 앱의 같은 자리에 실물을
띄운다. 중앙 위는 젯슨 카메라가 지각에 넘긴 프레임, 우측은 젯슨 온보드 SW 의 조회다.

`Sim` 자리에 들어간다. 앱이 `Sim` 에서 꺼내 쓰는 것 중 보는 것과 수동 조작만 낸다. 수동 · 자율은
젯슨이 쥐고 조회로 알려 준다 — 앱은 바꿔 달라고 보내고 조회가 바뀌면 따른다. 외부 시점은 두지 아니한다.

젯슨 송출은 젯슨 안쪽 주소에만 열린다. 여기서 ssh 터널을 띄워 붙는다.

실물 디버깅 도구(사양 4-5)가 보내는 명령과 주입 소리, 수동 조작의 키도 같은 연결로 간다.
"""

from __future__ import annotations

import json
import pickle
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import runtime as _runtime
from debug_link import recv_msg, send_msg
from panels import History, sections, summary_rows
from runtime import load_home_pose
from views import Views
from world import World

_실물_설정 = Path(_runtime.__file__).resolve().parents[1] / "real" / "config.json"
# 붙지 못했을 때 다시 붙는 간격이다.
_다시_붙기_s = 1.0
# 키가 그대로여도 이 간격으로 다시 보낸다. 젯슨은 0.3 초 동안 안 오면 놓은 것으로 친다 (Runtime `real/manual.py`).
_키_간격_s = 0.1


class Real:
    """젯슨 송출에 붙어 최신 조회와 프레임을 쥔다. 창을 알지 못한다."""

    real = True

    def __init__(self, host: str | None = "jetson", port: int | None = None) -> None:
        설정 = json.loads(_실물_설정.read_text(encoding="utf-8"))
        self.port = int(port if port is not None else 설정["link"]["port"])
        카메라 = 설정["camera"]

        # 그림 크기만 쓴다. 외부 시점을 그리지 아니한다.
        self.world = World()
        self.views = Views(
            self.world.model, robot_size=(int(카메라["out_width"]), int(카메라["out_height"])),
        )
        self.joint_names, _ = load_home_pose()

        self.runtime = self
        self.manual = False
        self.injector = None
        self.policy = Path("-")
        self.t = 0.0
        self.cycles = 0
        self.missed = 0
        self.cycle_ms = 0.0
        self.state = "연결 중"
        self._snapshot = None
        self._frame: np.ndarray | None = None
        self.history = History()

        self._자물쇠 = threading.Lock()
        self._보내기_자물쇠 = threading.Lock()
        self._새_조회: bytes | None = None
        self._새_프레임: bytes | None = None
        self._멈춤 = threading.Event()
        self._sock: socket.socket | None = None
        self._보낸_키: dict | None = None
        self._키_때 = 0.0
        self._터널 = None if host is None else subprocess.Popen(
            ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=5",
             "-L", f"{self.port}:127.0.0.1:{self.port}", host],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._실 = threading.Thread(target=self._받는다, name="실물 받기", daemon=True)
        self._실.start()

    # --- 받는 실 ---

    def _받는다(self) -> None:
        while not self._멈춤.is_set():
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
            except OSError:
                self.state = "대기"
                self._멈춤.wait(_다시_붙기_s)
                continue
            self._sock = sock
            sock.settimeout(5.0)
            인사 = False
            try:
                while not self._멈춤.is_set():
                    종류, 본문 = recv_msg(sock)
                    with self._자물쇠:
                        if 종류 == b"H":
                            인사 = True
                            self.policy = Path(json.loads(본문)["policy"])
                            self.state = "받는 중"
                        elif 종류 == b"S":
                            self._새_조회 = 본문
                        elif 종류 == b"F":
                            self._새_프레임 = 본문
                    if 종류 == b"F":
                        # 받았다고 알려야 젯슨이 다음 프레임을 보낸다 (Runtime `debug_link.py`).
                        with self._보내기_자물쇠:
                            send_msg(sock, b"A", b"")
            except (OSError, ConnectionError, struct.error) as e:
                # 터널은 젯슨 쪽이 안 열려 있어도 받아 준 뒤 끊는다. 인사 없이 끊기면 그것이다.
                self.state = f"끊김 ({type(e).__name__}) · 다시 연결 중" if 인사 else "대기"
            finally:
                sock.close()
                self._sock = None
            self._멈춤.wait(_다시_붙기_s)

    # --- 앱이 부르는 것 ---

    def due(self, elapsed_s: float) -> int:
        """새 조회가 왔으면 1 이다. 시계는 젯슨이 쥔다."""
        return 1 if self._새_조회 is not None else 0

    def cycle(self, ticks: int = 1) -> None:
        """받아 둔 최신 조회를 꺼내 시계를 맞춘다. 사이의 것은 버린다."""
        with self._자물쇠:
            본문, self._새_조회 = self._새_조회, None
        if 본문 is None:
            return
        try:
            조회, 덧 = pickle.loads(본문)
        except Exception:  # noqa: BLE001 — 덮어쓰는 중에 읽힌 칸이다. 버린다
            return
        self._snapshot = 조회
        self.history.update(조회)
        self.t = float(덧["t"])
        self.cycles = int(덧["k"])
        self.missed = int(덧["late"])
        self.cycle_ms = float(덧["cycle_ms"])
        self.manual = bool(덧.get("manual", False))

    @property
    def frame(self) -> np.ndarray | None:
        """젯슨 카메라가 지각에 넘긴 마지막 프레임이다. RGB."""
        with self._자물쇠:
            본문, self._새_프레임 = self._새_프레임, None
        if 본문 is not None:
            import cv2

            그림 = cv2.imdecode(np.frombuffer(본문, np.uint8, offset=8), cv2.IMREAD_COLOR)
            if 그림 is not None:
                self._frame = 그림
        return self._frame

    def snapshot(self):
        return self._snapshot

    def send_command(self, command: dict) -> bool:
        """디버깅 명령 하나를 젯슨 제어로 보낸다. 붙어 있지 않으면 거짓이다."""
        return self._보낸다(b"C", json.dumps(command, ensure_ascii=False).encode())

    def set_manual(self, on: bool) -> bool:
        """젯슨에 수동 · 자율을 바꿔 달라고 보낸다. `manual` 은 조회가 돌아올 때 바뀐다."""
        self._보낸_키 = None
        return self.send_command({"kind": "manual", "on": bool(on)})

    def drive(
        self, forward: float, lateral: float, turn: float,
        head_pitch: float = 0.0, head_yaw: float = 0.0, head_home: bool = False,
        ear: tuple[bool, bool] = (False, False),
    ) -> None:
        """키 누름 상태를 젯슨에 보낸다. 바뀌었거나 `_키_간격_s` 가 지났을 때만 보낸다."""
        if not self.manual:
            return
        키 = {"forward": forward, "lateral": lateral, "turn": turn, "head_pitch": head_pitch,
             "head_yaw": head_yaw, "head_home": head_home, "ear": list(ear)}
        지금 = time.monotonic()
        if 키 == self._보낸_키 and 지금 - self._키_때 < _키_간격_s:
            return
        if self.send_command({"kind": "keys", **키}):
            self._보낸_키, self._키_때 = 키, 지금

    def send_mic(self, pcm: np.ndarray) -> bool:
        """마이크 형(int16 모노 48 kHz) 소리를 젯슨 마이크에 섞게 보낸다. 붙어 있지 않으면 거짓이다."""
        return self._보낸다(b"M", np.asarray(pcm, dtype=np.int16).astype("<i2").tobytes())

    def _보낸다(self, kind: bytes, body: bytes) -> bool:
        sock = self._sock
        if sock is None or self.state != "받는 중":
            return False
        try:
            with self._보내기_자물쇠:
                send_msg(sock, kind, body)
        except OSError:
            return False
        return True

    def status(self):
        return sections(self._snapshot, self.joint_names)

    def summary(self):
        """우측 패널 맨 위 요약이다. 시뮬과 같은 꼴이다 (`panels.summary_rows`)."""
        return summary_rows(self._snapshot)

    def close(self) -> None:
        self._멈춤.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._실.join(timeout=3.0)
        if self._터널 is not None:
            self._터널.terminate()
            self._터널.wait(timeout=3.0)
        self.views.close()
