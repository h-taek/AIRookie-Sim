"""실물 실행기 (사양 4-4).

젯슨의 실물 진입점(Runtime `real/run.py`)을 앱에서 띄우고 멈춘다. 창을 알지 못한다.

**실행은 앱과 떨어져 돈다.** 젯슨에서 `setsid nohup` 으로 띄우고 출력은 젯슨의 파일에 쓴다.
앱이 죽거나 와이파이가 끊겨도 로봇은 그대로 돈다. 앱은 그 파일을 따라 읽는다. 터미널에서
띄운 실행도 도는지는 보인다.

**경로는 앱에서 고른 하나다.** '같은 망' 이면 맥과 같은 망의 이웃 중 젯슨 호스트 키가 맞는 주소를 찾아
그 주소로만 붙는다 -- 젯슨은 붙는 와이파이에 따라 주소가 바뀌므로 저장된 값을 믿지 아니한다. MAC 으로는
찾지 아니한다. macOS 가 이 앱의 파이썬이 부른 ARP · NDP 표에서 이웃을 가린다. 'NetBird' 면 ssh 별칭
(NetBird 주소)으로만 붙는다. 한쪽이 안 될 때 다른 쪽으로 넘어가지 아니한다. 같은 망은 인터넷을 거치지
않아 전시장처럼 인터넷이 불안한 곳에서도 붙는다. 송출 터널도 붙은 뒤에 연다. 젯슨이 꺼져 있으면 찾을
때까지 다시 찾는다. 전부 딴 실에서 하고 창을 막지 아니한다.

**멈춤은 Ctrl-C 와 같다.** SIGTERM 을 보내면 실물 진입점이 Ctrl-C 와 같은 길로 홈 자세 목표를 보내고
토크를 잡은 채 끝난다. SIGINT 는 쓰지 아니한다 — 셸이 뒤에 띄운 프로세스는 SIGINT 를 무시한다.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 젯슨 저장소 자리. 정책만 고르고 시범 동작은 지금 실물이 쓰는 것으로 둔다. 홈 자세는 src/config/home_pose.json 이다.
젯슨_저장소 = "~/dev/AIRookie_Runtime"
정책_폴더 = "assets"
시범_동작 = "assets/polynomial_coefficients.pkl"
# 앱이 띄운 실행의 출력 파일. 실행마다 덮어쓴다.
출력_파일 = "logs/dashboard_run.out"
# 도는 실물 진입점을 찾는 무늬. 터미널에서 띄운 것도 걸린다. 첫 글자를 [] 로 싸서 이 무늬를
# 담은 셸 명령 자체는 걸리지 않게 한다.
_무늬 = "[p]ython -m real[.]run"
# 젯슨 상태를 다시 묻는 간격, 못 찾았을 때 다시 찾는 간격이다.
_묻기_간격_s = 2.0
_다시_찾기_s = 5.0
# 맥 전사 서버(Runtime tools/asr_mac_server.py)의 기본 포트.
맥_전사_포트 = 8765
# 맥 전사 서버를 띄우는 자리와 명령. Runtime 저장소는 이 저장소 옆에 놓인다.
_런타임 = Path(__file__).resolve().parents[2] / "AIRookie_Runtime"
_전사_서버_명령 = ["run", "--no-project", "--python", "3.12", "--with", "mlx-audio==0.6.4",
                 "python", "tools/asr_mac_server.py", "--port", str(맥_전사_포트)]
# 맥 감정 서버(Runtime tools/emotion_mac_server.py)의 기본 포트. 감정 의존성이 이 앱의 venv 에 있어 그것으로 띄운다.
맥_감정_포트 = 8766
_감정_서버_명령 = [sys.executable, "tools/emotion_mac_server.py", "--port", str(맥_감정_포트)]
_ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3"]


def run_command(policy: str, arm: bool, safety: bool, no_servo: bool = False, servo_speed: int = 0,
                asr: str = "", asr_url: str = "", emotion: str = "", emotion_url: str = "") -> str:
    """젯슨에서 돌릴 셸 명령. 이미 돌고 있으면 띄우지 않고 그렇다고 낸다.

    `no_servo` 는 개발용이다. 서보 전원 없이 카메라 · 마이크 · 모델을 볼 때 쓴다.
    `servo_speed` 는 서보 속도 상한 레지스터(46) 값이다. 0 이면 넘기지 않는다(상한 없음).
    `asr` 는 전사 백엔드(Runtime `--asr`)다. 비우면 넘기지 않아 젯슨 autonomy.json 값을 쓴다.
    mac 이면 `asr_url`(맥 전사 서버 주소)을 함께 넘긴다. `emotion` · `emotion_url` 은 감정 추론 자리
    (Runtime `--emotion` · `--emotion-url`)로 같은 규칙이다.
    """
    인자 = ["--policy", f"{정책_폴더}/{policy}", "--reference", 시범_동작]
    if arm:
        인자.append("--arm")
    if safety:
        인자.append("--safety")
    if no_servo:
        인자.append("--no-servo")
    if servo_speed > 0:
        인자 += ["--servo-speed", str(int(servo_speed))]
    if asr:
        인자 += ["--asr", asr]
        if asr == "mac" and asr_url:
            인자 += ["--asr-url", asr_url]
    if emotion:
        인자 += ["--emotion", emotion]
        if emotion == "mac" and emotion_url:
            인자 += ["--emotion-url", emotion_url]
    # 모듈 이름을 따옴표로 싼다. 이 명령을 도는 셸의 명령 줄이 무늬에 걸리지 않게 한다.
    실행 = (
        f'setsid nohup env PYTHONPATH=src:. PYTHONUNBUFFERED=1 .venv/bin/python -m "real.run" '
        f"{shlex.join(인자)} > {출력_파일} 2>&1 < /dev/null &"
    )
    return f"cd {젯슨_저장소} && if pgrep -f {shlex.quote(_무늬)} > /dev/null; then echo 이미 실행 중; else {실행} echo 실행함; fi"


def stop_command() -> str:
    return f"pkill -TERM -f {shlex.quote(_무늬)} && echo 정지 요청함 || echo 실행 중 아님"


def _state_command() -> str:
    return f"cd {젯슨_저장소} && (pgrep -f {shlex.quote(_무늬)} > /dev/null && echo RUN || echo IDLE); ls {정책_폴더}"


def _응답한_주소(ping6_글: str) -> list[str]:
    """`ping6 ff02::1%en0` 출력에서 답한 링크 로컬 주소(`%en0` 붙음)를 고른다. 맥 자신(보낸 주소)은 버린다.

    링크 로컬만 쓴다 -- 전역 주소는 젯슨이 수시로 바꾼다.
    """
    보낸 = re.search(r"\) (\S+) -->", ping6_글)
    주소들 = []
    for 주소 in re.findall(r"bytes from (\S+?),", ping6_글):
        if 주소 not in 주소들 and not (보낸 and 주소 == 보낸.group(1)):
            주소들.append(주소)
    return 주소들


def _키_이름(host: str) -> str:
    """ssh 별칭이 가리키는 주소(NetBird). known_hosts 에 젯슨 호스트 키가 이 이름으로 올라 있다."""
    for 줄 in subprocess.run(["ssh", "-G", host], capture_output=True, text=True).stdout.splitlines():
        if 줄.startswith("hostname "):
            return 줄.split()[1]
    return host


def _와이파이_주소(ifconfig_글: str) -> str:
    """`ifconfig en0` 출력에서 이 맥의 주소를 고른다. 없으면 빈 글이다.

    IPv4 가 있으면 그것이다. 아이폰 핫스팟은 IPv4 를 주지 않고 IPv6 만 준다 -- 그때 보이는
    192.0.0.x 는 macOS 가 IPv4 를 흉내 내는 주소(CLAT)라 밖에서 닿지 않으므로 버리고, 고정 전역
    IPv6 를 쓴다. 임시(temporary) · 흉내(clat46) · 링크 로컬 주소는 고르지 아니한다.
    """
    v4, v6 = [], []
    for 줄 in ifconfig_글.splitlines():
        칸 = 줄.split()
        if len(칸) < 2:
            continue
        if 칸[0] == "inet" and not 칸[1].startswith(("192.0.0.", "169.254.", "127.")):
            v4.append(칸[1])
        elif (칸[0] == "inet6" and not 칸[1].lower().startswith("fe80")
              and not {"temporary", "clat46", "deprecated"} & set(칸)):
            v6.append(칸[1])
    return (v4 or v6 or [""])[0]


def _맥_주소() -> str:
    """이 맥의 와이파이(en0) 주소. IPv4 가 없는 망(아이폰 핫스팟)이면 전역 IPv6 다. 없으면 빈 글이다."""
    return _와이파이_주소(subprocess.run(["ifconfig", "en0"], capture_output=True, text=True).stdout)


def _넷버드_주소_고른다(ifconfig_글: str) -> str:
    """`ifconfig` 출력에서 NetBird 주소(100.64.0.0/10)를 고른다. 없으면 빈 글이다."""
    for 줄 in ifconfig_글.splitlines():
        칸 = 줄.split()
        if len(칸) > 1 and 칸[0] == "inet" and 칸[1].startswith("100."):
            if 64 <= int(칸[1].split(".")[1]) <= 127:
                return 칸[1]
    return ""


def _넷버드_주소() -> str:
    """이 맥의 NetBird 주소. 없으면 빈 글이다."""
    return _넷버드_주소_고른다(subprocess.run(["ifconfig"], capture_output=True, text=True).stdout)


def asr_url(값: str = "", route: str = "같은 망", port: int = 맥_전사_포트) -> str:
    """맥 서버 주소. 앱과 전사 · 감정 서버가 같은 맥에서 돌므로 이 맥의 주소다.

    젯슨이 이 맥에 닿는 길은 젯슨에 붙은 경로와 같다 -- 'NetBird' 면 이 맥의 NetBird 주소, '같은 망' 이면
    와이파이 주소다. 포트는 전사 서버면 `맥_전사_포트`, 감정 서버면 `맥_감정_포트` 다. IPv6 주소는 대괄호로 싼다.
    주소가 없으면 빈 글이다. `값` 은 시험이 주소를 정해 넣는 자리다.
    """
    if 값.strip():
        return 값.strip()
    맥 = _넷버드_주소() if route == "NetBird" else _맥_주소()
    if not 맥:
        return ""
    return f"http://[{맥}]:{port}" if ":" in 맥 else f"http://{맥}:{port}"


def start_asr_server() -> subprocess.Popen | None:
    """맥 전사 서버를 띄운다. 그 포트가 이미 열려 있으면(먼저 띄운 서버) 띄우지 않고 없음을 낸다.

    출력은 앱 설정 자리(`~/.config/airookie_sim/asr_mac_server.log`)에 쓴다. 모델은 서버가 시작할 때 올린다(1 초 안쪽).
    `uv` 가 없으면 띄우지 못한다고 찍고 없음을 낸다 -- 그때 맥 Qwen 은 대비 전사로 돈다.
    """
    uv = shutil.which("uv")
    if uv is None:
        print("[전사 서버] uv 를 찾지 못해 띄우지 못했다")
        return None
    return _서버를_띄운다("전사 서버", 맥_전사_포트, [uv, *_전사_서버_명령], "asr_mac_server.log")


def start_emotion_server() -> subprocess.Popen | None:
    """맥 감정 서버(em7)를 띄운다. 규칙은 `start_asr_server` 와 같다. 기록은 `emotion_mac_server.log` 다."""
    return _서버를_띄운다("감정 서버", 맥_감정_포트, _감정_서버_명령, "emotion_mac_server.log")


def _서버를_띄운다(이름: str, 포트: int, 명령: list[str], 기록_이름: str) -> subprocess.Popen | None:
    with socket.socket() as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", 포트)) == 0:
            print(f"[{이름}] {포트} 포트에 이미 떠 있다. 그것을 쓴다")
            return None
    자리 = Path.home() / ".config" / "airookie_sim" / 기록_이름
    자리.parent.mkdir(parents=True, exist_ok=True)
    기록 = open(자리, "w")
    print(f"[{이름}] 띄움 — 기록 {자리}")
    # 제 프로세스 묶음으로 띄운다. 내릴 때 uv 와 그 아래 파이썬을 함께 내린다.
    return subprocess.Popen(명령, cwd=_런타임, stdin=subprocess.DEVNULL,
                            stdout=기록, stderr=subprocess.STDOUT, start_new_session=True)


def stop_asr_server(서버: subprocess.Popen | None) -> None:
    """`start_asr_server` · `start_emotion_server` 가 띄운 서버를 내린다. 먼저 떠 있던 서버(없음)는 건드리지 아니한다."""
    if 서버 is None or 서버.poll() is not None:
        return
    os.killpg(서버.pid, signal.SIGTERM)
    try:
        서버.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(서버.pid, signal.SIGKILL)


def find_candidates() -> list[str]:
    """같은 망에서 젯슨일 수 있는 주소들이다. 어느 것이 젯슨인지는 ssh 가 호스트 키로 가린다.

    IPv4 망이면 맥의 /24 망을 한 번씩 두드려 답한 주소를 모은다. 링크의 모든 기기를 부르는 ff02::1 에
    답한 IPv6 주소도 모은다 -- 아이폰 핫스팟은 IPv6 만 준다. 이웃 표(ARP · NDP)는 보지 아니한다. 핫스팟은
    젯슨 응답이 0.5 초까지 늦어 1 초 넘게 기다린다.
    """
    후보 = []
    맥 = _맥_주소()
    if 맥 and ":" not in 맥:
        앞 = 맥.rsplit(".", 1)[0]
        주소들 = [f"{앞}.{i}" for i in range(1, 255) if f"{앞}.{i}" != 맥]
        with ThreadPoolExecutor(max_workers=64) as 일꾼:
            답 = list(일꾼.map(lambda a: subprocess.run(["ping", "-c1", "-W", "1000", a],
                                                         capture_output=True).returncode == 0, 주소들))
        후보 = [a for a, 됨 in zip(주소들, 답) if 됨]
    글 = subprocess.run(["ping6", "-c", "3", "-i", "0.5", "ff02::1%en0"], capture_output=True, text=True).stdout
    return 후보 + _응답한_주소(글)


class Launcher:
    """젯슨을 찾아 붙고, 송출 터널을 열고, 실물 진입점을 띄우고 멈춘다. ssh 는 전부 딴 실에서 부른다."""

    def __init__(self, host: str = "jetson", tunnel_port: int | None = None, route: str = "같은 망") -> None:
        self.host = host
        self.tunnel_port = tunnel_port
        self.route = route
        self.address: str | None = None
        self.connected = False
        self.running = False
        self.policies: list[str] = []
        self.note = "찾는 중"
        self.lines: deque[str] = deque(maxlen=200)
        self._주소_인자: list[str] = []
        self._멈춤 = threading.Event()
        self._꼬리: subprocess.Popen | None = None
        self._터널: subprocess.Popen | None = None
        threading.Thread(target=self._돈다, name="실행기", daemon=True).start()

    def _ssh_cmd(self, *extra: str) -> list[str]:
        return [*_ssh, *self._주소_인자, *extra, self.host]

    def _ssh(self, command: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
        return subprocess.run([*self._ssh_cmd(), command], capture_output=True, text=True, timeout=timeout)

    def _붙어_본다(self, 인자: list[str]) -> bool:
        try:
            return subprocess.run([*_ssh, *인자, self.host, "true"], capture_output=True, timeout=6.0).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _찾는다(self) -> bool:
        """고른 경로로만 붙는다. 같은 망이면 이웃 중 젯슨 호스트 키가 맞는 주소로, NetBird 면 별칭으로.

        같은 망 주소에는 HostKeyAlias 로 별칭의 키를 대어 본다. 키가 다른 기기에는 붙지 않고, 젯슨 주소가
        바뀌어도 known_hosts 를 고치지 않는다. 링크 로컬의 `%` 는 ssh 가 바꿈 글자로 읽으므로 `%%` 로 넘긴다.
        """
        if self.route == "NetBird":
            self._주소_인자 = []
            self.note = "NetBird 로 붙는 중"
            if self._붙어_본다([]):
                self.address = f"{self.host} · NetBird"
                return True
            self.note = "NetBird 로 못 붙음 · 전원·NetBird 확인"
            return False
        self.note = "같은 망에서 찾는 중"
        후보 = find_candidates()
        if not 후보:
            self.note = "같은 망에서 못 찾음 · 전원·와이파이 확인"
            return False
        키 = _키_이름(self.host)
        인자들 = [["-o", f"HostName={주소.replace('%', '%%')}", "-o", f"HostKeyAlias={키}"] for 주소 in 후보]
        with ThreadPoolExecutor(max_workers=16) as 일꾼:
            붙음 = list(일꾼.map(self._붙어_본다, 인자들))
        for 주소, 인자, 됨 in zip(후보, 인자들, 붙음):
            if 됨:
                self._주소_인자 = 인자
                self.address = f"{주소} · 같은 망"
                return True
        self.note = f"같은 망 이웃 {len(후보)}곳 중 젯슨 키가 맞는 곳 없음"
        return False

    def set_route(self, route: str) -> None:
        """경로를 바꾼다. 붙어 있던 연결 · 송출 터널 · 출력 읽기를 끊어 다음 바퀴에 새 경로로 찾게 한다."""
        if route == self.route:
            return
        self.route = route
        self.connected = False
        self.note = f"{route} 로 다시 찾는 중"
        for 것 in (self._꼬리, self._터널):
            if 것 is not None and 것.poll() is None:
                것.terminate()
        self._터널 = None

    def _돈다(self) -> None:
        while not self._멈춤.is_set():
            if not self.connected:
                if not self._찾는다():
                    self._멈춤.wait(_다시_찾기_s)
                    continue
                self.connected = True
                self.note = f"연결됨 · {self.address}"
                threading.Thread(target=self._따라_읽는다, name="실행기 출력", daemon=True).start()
            self._터널을_잇는다()
            try:
                r = self._ssh(_state_command())
            except subprocess.TimeoutExpired:
                r = None
            if r is None or r.returncode != 0:
                # 끊겼다. 주소가 바뀌었을 수 있으니 처음부터 찾는다.
                self.connected = False
                self.note = "끊김 · 다시 찾는 중"
                continue
            줄 = r.stdout.split()
            self.running = 줄[0] == "RUN"
            self.policies = sorted(p for p in 줄[1:] if p.endswith(".onnx"))
            self._멈춤.wait(_묻기_간격_s)

    def _터널을_잇는다(self) -> None:
        """송출 터널이 없거나 죽었으면 연다. 젯슨 쪽 송출이 아직 없어도 터널은 열어 둔다."""
        if self.tunnel_port is None or (self._터널 is not None and self._터널.poll() is None):
            return
        포트 = self.tunnel_port
        self._터널 = subprocess.Popen(
            [*self._ssh_cmd("-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=5",
                            "-L", f"{포트}:127.0.0.1:{포트}")],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _따라_읽는다(self) -> None:
        """출력 파일을 따라 읽는다. 끊기면 붙어 있는 동안 다시 붙는다. 파일이 새로 써지면 tail -F 가 따라간다."""
        while not self._멈춤.is_set() and self.connected:
            self._꼬리 = subprocess.Popen(
                [*self._ssh_cmd(), f"tail -n 40 -F {젯슨_저장소}/{출력_파일} 2>/dev/null"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            for 줄 in self._꼬리.stdout:
                self.lines.append(줄.rstrip("\n"))
            self._꼬리.wait()
            self._멈춤.wait(_묻기_간격_s)

    def _딴_실에서(self, command: str) -> None:
        if not self.connected:
            self.note = "연결 안 됨"
            return

        def 돌린다():
            try:
                r = self._ssh(command)
                self.note = (r.stdout.strip() or r.stderr.strip())[-120:]
            except subprocess.TimeoutExpired:
                self.note = "응답 없음"

        threading.Thread(target=돌린다, daemon=True).start()

    def start(self, policy: str, arm: bool, safety: bool, no_servo: bool = False, servo_speed: int = 0,
              asr: str = "", asr_url: str = "", emotion: str = "", emotion_url: str = "") -> None:
        self.note = "실행 요청 중"
        self._딴_실에서(run_command(policy, arm, safety, no_servo, servo_speed, asr, asr_url, emotion, emotion_url))

    def stop(self) -> None:
        self.note = "정지 요청 중"
        self._딴_실에서(stop_command())

    def close(self) -> None:
        """앱 쪽만 정리한다. 젯슨에서 도는 실행은 건드리지 아니한다."""
        self._멈춤.set()
        for 것 in (self._꼬리, self._터널):
            if 것 is not None and 것.poll() is None:
                것.terminate()
