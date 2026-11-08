"""실물 실행기 (launcher.py) 의 젯슨 찾기 시험. ssh 와 망 훑기는 가짜로 바꾼다."""

from __future__ import annotations

import launcher


_키 = "100.85.149.201"


def _인자(주소: str) -> list[str]:
    return ["-o", f"HostName={주소}", "-o", f"HostKeyAlias={_키}"]


def _실행기(monkeypatch, 경로: str, 후보: list[str], 붙는_인자: list[list[str]]):
    """실을 띄우지 않은 실행기. `붙는_인자` 로 시도한 ssh 만 붙는다. 망 훑기 횟수와 시도마다 쓴 주소 인자를 남긴다."""
    훑음: list[int] = []

    def 찾는다():
        훑음.append(1)
        return 후보

    monkeypatch.setattr(launcher, "find_candidates", 찾는다)
    monkeypatch.setattr(launcher, "_키_이름", lambda host: _키)
    실행기 = launcher.Launcher.__new__(launcher.Launcher)
    실행기.host, 실행기.route, 실행기.address, 실행기.note, 실행기._주소_인자 = "jetson", 경로, None, "", []
    시도: list[list[str]] = []

    def 붙어_본다(인자: list[str]) -> bool:
        시도.append(list(인자))
        return 인자 in 붙는_인자

    실행기._붙어_본다 = 붙어_본다
    return 실행기, 시도, 훑음


def test_같은_망을_고르면_젯슨_키가_맞는_주소로만_붙는다(monkeypatch):
    실행기, 시도, _ = _실행기(monkeypatch, "같은 망", ["172.20.10.1", "172.20.10.3"], [_인자("172.20.10.3")])
    assert 실행기._찾는다()
    assert sorted(시도) == sorted([_인자("172.20.10.1"), _인자("172.20.10.3")])
    assert 실행기._주소_인자 == _인자("172.20.10.3")
    assert 실행기.address == "172.20.10.3 · 같은 망"


def test_링크_로컬_주소의_퍼센트는_ssh_에_두_번_넘긴다(monkeypatch):
    링크 = "fe80::473e:81a9:8251:9541%en0"
    실행기, _, _ = _실행기(monkeypatch, "같은 망", [링크], [_인자("fe80::473e:81a9:8251:9541%%en0")])
    assert 실행기._찾는다()
    assert 실행기.address == f"{링크} · 같은 망"


def test_같은_망을_고르고_못_찾으면_NetBird_로_넘어가지_않는다(monkeypatch):
    실행기, 시도, _ = _실행기(monkeypatch, "같은 망", [], [])
    assert not 실행기._찾는다()
    assert 시도 == []
    assert "같은 망" in 실행기.note


def test_같은_망_이웃_어디도_키가_맞지_않아도_NetBird_로_넘어가지_않는다(monkeypatch):
    실행기, 시도, _ = _실행기(monkeypatch, "같은 망", ["192.168.0.34"], [])
    assert not 실행기._찾는다()
    assert 시도 == [_인자("192.168.0.34")]
    assert "같은 망" in 실행기.note


def test_NetBird_를_고르면_망을_훑지_않고_별칭으로만_붙는다(monkeypatch):
    실행기, 시도, 훑음 = _실행기(monkeypatch, "NetBird", ["172.20.10.3"], [[]])
    assert 실행기._찾는다()
    assert 시도 == [[]] and 훑음 == []
    assert 실행기.address == "jetson · NetBird"


def test_NetBird_가_안_되면_못_찾았다고_알린다(monkeypatch):
    실행기, 시도, _ = _실행기(monkeypatch, "NetBird", [], [])
    assert not 실행기._찾는다()
    assert "NetBird" in 실행기.note


def test_경로를_바꾸면_연결을_풀고_터널과_출력_읽기를_끊는다():
    class 가짜_프로세스:
        def __init__(self):
            self.끊김 = False

        def poll(self):
            return 0 if self.끊김 else None

        def terminate(self):
            self.끊김 = True

    실행기 = launcher.Launcher.__new__(launcher.Launcher)
    터널, 꼬리 = 가짜_프로세스(), 가짜_프로세스()
    실행기.route, 실행기.connected, 실행기._터널, 실행기._꼬리, 실행기.note = "같은 망", True, 터널, 꼬리, ""
    실행기.set_route("NetBird")
    assert 실행기.route == "NetBird" and not 실행기.connected
    assert 터널.끊김 and 꼬리.끊김 and 실행기._터널 is None


def test_같은_경로를_다시_고르면_연결을_건드리지_않는다():
    실행기 = launcher.Launcher.__new__(launcher.Launcher)
    실행기.route, 실행기.connected = "NetBird", True
    실행기.set_route("NetBird")
    assert 실행기.connected


def test_맥_전사를_고르면_실행_명령에_전사_인자를_싣는다():
    명령 = launcher.run_command("v2_48_300M.onnx", True, False, asr="mac", asr_url="http://172.20.10.4:8765")
    assert "--asr mac --asr-url http://172.20.10.4:8765" in 명령
    assert "--asr" not in launcher.run_command("v2_48_300M.onnx", True, False)


def test_맥_주소를_비우면_이_맥의_와이파이_주소로_채운다(monkeypatch):
    monkeypatch.setattr(launcher, "_맥_주소", lambda: "172.20.10.4")
    assert launcher.asr_url("") == "http://172.20.10.4:8765"
    assert launcher.asr_url("  ") == "http://172.20.10.4:8765"
    assert launcher.asr_url("http://10.0.0.9:8765") == "http://10.0.0.9:8765"


def test_와이파이_주소가_없으면_빈_주소를_낸다(monkeypatch):
    monkeypatch.setattr(launcher, "_맥_주소", lambda: "")
    assert launcher.asr_url("") == ""


# 아이폰 핫스팟(IPv6 만 준다)에서 본 그대로다. 192.0.0.2 는 macOS 의 IPv4 흉내(CLAT) 주소다.
_핫스팟_ifconfig = """\
\tinet6 fe80::10cb:7842:ce15:848b%en0 prefixlen 64 secured scopeid 0xf
\tinet6 2001:e60:a574:d6a7:86:e9c3:b158:c4f0 prefixlen 64 autoconf secured
\tinet6 2001:e60:a574:d6a7:160:a43d:80d0:21f5 prefixlen 64 autoconf temporary
\tinet 192.0.0.2 netmask 0xffffffff broadcast 192.0.0.2
\tinet6 2001:e60:a574:d6a7:cf5:ffa9:435b:3c08 prefixlen 64 clat46
"""


def test_핫스팟이면_IPv4_흉내를_버리고_고정_전역_IPv6_를_맥_주소로_쓴다():
    assert launcher._와이파이_주소(_핫스팟_ifconfig) == "2001:e60:a574:d6a7:86:e9c3:b158:c4f0"


def test_IPv4_가_있으면_IPv4_를_맥_주소로_쓴다():
    글 = "\tinet6 fe80::1%en0 prefixlen 64\n\tinet 192.168.0.34 netmask 0xffffff00 broadcast 192.168.0.255\n"
    assert launcher._와이파이_주소(글) == "192.168.0.34"


def test_IPv6_맥_주소는_대괄호로_싼다(monkeypatch):
    monkeypatch.setattr(launcher, "_맥_주소", lambda: "2001:e60::5")
    assert launcher.asr_url("") == "http://[2001:e60::5]:8765"


def test_ping6_에_답한_링크_로컬_주소를_맥_자신을_빼고_한_번씩_고른다():
    # 아이폰 핫스팟에서 본 그대로다.
    글 = """\
PING6(56=40+8+8 bytes) fe80::10cb:7842:ce15:848b%en0 --> ff02::1%en0
16 bytes from fe80::10cb:7842:ce15:848b%en0, icmp_seq=0 hlim=64 time=0.129 ms
16 bytes from fe80::b455:75ff:fe09:8064%en0, icmp_seq=0 hlim=64 time=72.014 ms
16 bytes from fe80::473e:81a9:8251:9541%en0, icmp_seq=0 hlim=64 time=227.613 ms
16 bytes from fe80::10cb:7842:ce15:848b%en0, icmp_seq=1 hlim=64 time=0.173 ms
16 bytes from fe80::473e:81a9:8251:9541%en0, icmp_seq=1 hlim=64 time=73.257 ms

--- ff02::1%en0 ping6 statistics ---
"""
    assert launcher._응답한_주소(글) == ["fe80::b455:75ff:fe09:8064%en0", "fe80::473e:81a9:8251:9541%en0"]


def test_전사_서버가_이미_떠_있으면_새로_띄우지_않는다():
    import socket

    with socket.socket() as 서버:
        서버.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            서버.bind(("127.0.0.1", launcher.맥_전사_포트))
        except OSError:
            pass  # 이미 누가 쥐고 있다 -- 그것도 '떠 있다' 이다
        else:
            서버.listen()
        assert launcher.start_asr_server() is None


def test_띄운_적_없는_서버는_내리지_않는다():
    launcher.stop_asr_server(None)


def test_NetBird_경로면_맥의_NetBird_주소를_전사_서버_주소로_쓴다(monkeypatch):
    monkeypatch.setattr(launcher, "_넷버드_주소", lambda: "100.85.176.239")
    monkeypatch.setattr(launcher, "_맥_주소", lambda: "2001:e60::5")
    assert launcher.asr_url(route="NetBird") == "http://100.85.176.239:8765"
    assert launcher.asr_url(route="같은 망") == "http://[2001:e60::5]:8765"


def test_ifconfig_에서_NetBird_주소를_고른다():
    글 = """utun3: flags=8051<UP>\n\tinet6 fe80::1%utun3 prefixlen 64\nutun100: flags=8051<UP> mtu 1280
\tinet 100.85.176.239 --> 100.85.176.239 netmask 0xffff0000\nen0: flags=8863<UP>\n\tinet 192.0.0.2 netmask 0xffffffff\n"""
    assert launcher._넷버드_주소_고른다(글) == "100.85.176.239"
    assert launcher._넷버드_주소_고른다("en0: flags\n\tinet 192.168.0.3 netmask\n") == ""


def test_맥_감정을_고르면_실행_명령에_감정_인자를_싣는다():
    명령 = launcher.run_command("v2_48_300M.onnx", True, False, emotion="mac",
                              emotion_url="http://172.20.10.4:8766")
    assert "--emotion mac --emotion-url http://172.20.10.4:8766" in 명령
    assert "--emotion" not in launcher.run_command("v2_48_300M.onnx", True, False)


def test_감정_서버_주소는_전사와_같은_경로의_맥_주소에_감정_포트다(monkeypatch):
    monkeypatch.setattr(launcher, "_넷버드_주소", lambda: "100.85.176.239")
    monkeypatch.setattr(launcher, "_맥_주소", lambda: "2001:e60::5")
    assert launcher.asr_url(route="NetBird", port=launcher.맥_감정_포트) == "http://100.85.176.239:8766"
    assert launcher.asr_url(route="같은 망", port=launcher.맥_감정_포트) == "http://[2001:e60::5]:8766"


def test_감정_서버_포트가_이미_열려_있으면_띄우지_않는다():
    import socket

    with socket.socket() as 서버:
        서버.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            서버.bind(("127.0.0.1", launcher.맥_감정_포트))
        except OSError:
            pass  # 이미 누가 쥐고 있다 -- 그것도 '떠 있다' 이다
        else:
            서버.listen()
        assert launcher.start_emotion_server() is None
