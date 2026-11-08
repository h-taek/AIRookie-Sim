"""카메라와 마이크에 흘려 넣을 입력원.

시뮬레이터는 검증 모드 둘로 돈다 (사양 3장).

| 모드 | 로봇 카메라 프레임 | 무엇을 본다 |
|---|---|---|
| 추종 | MuJoCo 렌더 | 사람을 따라가고 얼굴을 응시하는가 |
| 감정 표현 | 여기의 `FrameSource` (영상·웹캠) | 감정을 읽고 안 넘어지면서 동작하는가 |

**판정 결과를 넣지 아니한다** (사양 3-1). 넣는 것은 소리와 그림이고,
그것이 무슨 감정인지는 온보드 SW 가 정한다.

**직전 것을 다시 싣지 아니한다** (드라이버 계약 2-3). 카메라와 마이크는
제어 주기보다 느리게 돌므로 새 것이 없는 주기에는 `None` 을 낸다. 같은 것을
또 내보내면 읽는 쪽이 나이로 판정할 수 없다.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from speech import MIC_SAMPLE_RATE

# macOS 내장 음성이다. 한국어 음성 아홉 중 하나이며 `say -v '?'` 로 볼 수 있다.
기본_TTS_음성 = "Yuna"

# 마이크 자리에 싣는 표본율이다. 실물 마이크와 같다 (계약 2-3). 소리 입력원은 모두
# 이 표본율로 내고, 다른 표본율의 WAV 는 읽을 때 바꾼다.
기본_표본율 = MIC_SAMPLE_RATE

# 마이크가 들고 있을 소리의 상한이다. 앱 고리가 서 있는 동안에도(창을 끄는
# 동안 등) 마이크는 계속 들어오므로 상한이 없으면 메모리가 늘고, 고리가 다시
# 돌 때 한참 전 소리부터 나간다.
마이크_보관_s = 1.0


# --- 영상 클립 자료 ---


@dataclass(frozen=True)
class VideoClip:
    """영상 클립 하나. 소리와 동영상이 짝이다."""

    clip_id: str
    label: str
    text: str
    video: Path
    wav: Path
    source: str
    start_s: float
    end_s: float


def video_clips(폴더: Path | str) -> list[VideoClip]:
    """받아 둔 감정 영상 목록을 읽는다. 자료가 없으면 빈 목록이다.

    AI Hub 원본 영상에서 자른 클립이며 저장소에 담지 않는다.
    없다고 앱이 안 서면 안 된다.
    """
    폴더 = Path(폴더)
    목록 = 폴더 / "index.csv"
    if not 목록.exists():
        return []

    with 목록.open(encoding="utf-8", newline="") as f:
        return [
            VideoClip(
                clip_id=줄["clip_id"],
                label=줄["label"],
                text=줄["text"],
                video=폴더 / 줄["video"],
                wav=폴더 / 줄["wav"],
                source=줄["source"],
                start_s=float(줄["start_s"]),
                end_s=float(줄["end_s"]),
            )
            for 줄 in csv.DictReader(f)
        ]


# --- 그림 다루기 ---


def _resize(image: np.ndarray, 높이: int, 너비: int) -> np.ndarray:
    """최근방 이웃으로 크기를 바꾼다. 외부 라이브러리를 쓰지 아니한다."""
    원_높이, 원_너비 = image.shape[:2]
    if (원_높이, 원_너비) == (높이, 너비):
        return image

    행 = np.clip((np.arange(높이) * (원_높이 / 높이)).astype(int), 0, 원_높이 - 1)
    열 = np.clip((np.arange(너비) * (원_너비 / 너비)).astype(int), 0, 원_너비 - 1)
    return image[행[:, None], 열[None, :]]


def fit_frame(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """그림을 (너비, 높이) 화면에 레터박스로 넣는다.

    **늘려서 채우지 아니한다.** 정사각 얼굴을 2:1 화면에 늘리면 얼굴이 가로로
    퍼져 표정이 달라 보인다. 감정을 읽는 것이 목적이므로 비율을 지키고 남는
    자리는 검정으로 둔다.
    """
    목표_너비, 목표_높이 = int(size[0]), int(size[1])
    높이, 너비 = image.shape[:2]
    if (높이, 너비) == (목표_높이, 목표_너비):
        return image

    배율 = min(목표_너비 / 너비, 목표_높이 / 높이)
    새_너비 = max(1, int(round(너비 * 배율)))
    새_높이 = max(1, int(round(높이 * 배율)))

    화면 = np.zeros((목표_높이, 목표_너비, image.shape[2]), dtype=image.dtype)
    y0 = (목표_높이 - 새_높이) // 2
    x0 = (목표_너비 - 새_너비) // 2
    화면[y0 : y0 + 새_높이, x0 : x0 + 새_너비] = _resize(image, 새_높이, 새_너비)
    return 화면


# --- 프레임 입력원 ---


class FrameSource:
    """로봇 카메라 자리에 들어갈 그림을 내놓는 것."""

    def frame(self, t: float) -> np.ndarray | None:
        """새 프레임을 낸다. 새 것이 없으면 `None` 이다."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoFile(FrameSource):
    """동영상 파일에서 프레임을 읽어 내놓는다.

    클립이 3 ~ 6 초라 크지 않으므로 한 번에 다 읽어 두고 고리 안에서 디스크와
    디코딩을 하지 않게 한다.
    """

    def __init__(self, frames: list[np.ndarray], fps: float, loop: bool = True) -> None:
        if not frames:
            raise ValueError("프레임이 없다")
        if fps <= 0.0:
            raise ValueError(f"fps 가 0 이하다: {fps}")

        self._frames = frames
        self.fps = float(fps)
        self.loop = bool(loop)
        # 마지막으로 내보낸 칸이다. 직전과 같은 칸을 다시 싣지 않으려고 든다.
        self._앞_칸 = -1

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        size: tuple[int, int] | None = None,
        loop: bool = True,
    ) -> VideoFile:
        """mp4 등 영상 파일에서 만든다.

        **한 번에 다 읽어 둔다.** 고리 안에서 디스크·디코딩을 하지 않게 된다.
        OpenCV 로 읽고 RGB 로 바꾸며 fit_frame 으로 size 에 맞춘다.
        fps 는 파일에서 읽는다.
        """
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열지 못했다: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0:
            fps = 30.0

        frames = []
        try:
            while True:
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                # OpenCV 는 BGR 로 읽는다. 계약이 쓰는 것은 RGB 다.
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                frames.append(fit_frame(rgb, size) if size is not None else rgb)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"프레임이 없다: {path}")

        return cls(frames, fps, loop=loop)

    @property
    def duration_s(self) -> float:
        return len(self._frames) / self.fps

    def frame(self, t: float) -> np.ndarray | None:
        if t < 0.0:
            return None

        총_칸 = int(t * self.fps)
        if self.loop:
            칸 = 총_칸 % len(self._frames)
        else:
            if 총_칸 >= len(self._frames):
                return None
            칸 = 총_칸

        if 총_칸 == self._앞_칸:
            return None
        self._앞_칸 = 총_칸
        return self._frames[칸]

    def rewind(self) -> None:
        """처음부터 다시 내보낸다."""
        self._앞_칸 = -1


class Webcam(FrameSource):
    """맥북 카메라. 딴 실에서 받아 두고 물으면 최신 한 장을 넘긴다.

    **고리 안에서 직접 읽지 아니한다.** `VideoCapture.read` 는 다음 프레임이
    올 때까지 막는다. 30 fps 카메라면 한 번에 33 ms 까지 막히는데 제어 주기가
    20 ms 라 예산을 그것만으로 넘긴다.
    """

    def __init__(self, index: int = 0, size: tuple[int, int] = (960, 480)) -> None:
        import cv2

        self._cap = cv2.VideoCapture(int(index))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"카메라를 열지 못했다 (index={index}). "
                "시스템 설정 > 개인정보 보호 및 보안 > 카메라 를 확인한다"
            )

        self._size = (int(size[0]), int(size[1]))
        self._칸: np.ndarray | None = None
        self._자물쇠 = threading.Lock()
        self._멈춤 = threading.Event()
        self._실 = threading.Thread(target=self._돈다, name="webcam", daemon=True)
        self._실.start()

    def _돈다(self) -> None:
        while not self._멈춤.is_set():
            받음, bgr = self._cap.read()
            if not 받음:
                continue
            프레임 = fit_frame(np.ascontiguousarray(bgr[:, :, ::-1]), self._size)
            with self._자물쇠:
                self._칸 = 프레임

    def frame(self, t: float = 0.0) -> np.ndarray | None:
        """받아 둔 한 장을 넘기고 칸을 비운다. 비어 있으면 `None` 이다."""
        with self._자물쇠:
            프레임, self._칸 = self._칸, None
        return 프레임

    def close(self) -> None:
        self._멈춤.set()
        # 읽는 중이면 그 한 장이 끝나기를 기다린다. 기다리지 않고 풀어 버리면
        # 다른 실이 쓰는 중인 자원을 놓게 된다.
        self._실.join(timeout=2.0)
        self._cap.release()


# --- 소리 입력원 ---


class AudioSource:
    """마이크 자리에 흘려 넣을 소리를 내놓는 것."""

    sample_rate: int = 0

    def block(self, n: int) -> np.ndarray | None:
        """한 주기 분량 `n` 표본을 낸다. 낼 것이 없으면 `None` 이다."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """WAV 를 읽어 모노 float32 [-1, 1] 배열과 표본율을 낸다.

    표준 `wave` 모듈만 쓴다. 스테레오는 채널을 평균낸다.
    """
    with wave.open(str(path), "rb") as wf:
        채널 = wf.getnchannels()
        폭 = wf.getsampwidth()
        표본율 = wf.getframerate()
        원본 = wf.readframes(wf.getnframes())

    if 폭 == 1:
        자료 = (np.frombuffer(원본, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif 폭 == 2:
        자료 = np.frombuffer(원본, dtype=np.int16).astype(np.float32) / 32768.0
    elif 폭 == 3:
        raw = np.frombuffer(원본, dtype=np.uint8).reshape(-1, 3)
        int24 = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        int24 = np.where(int24 & 0x800000, int24 - 0x1000000, int24)
        자료 = int24.astype(np.float32) / 8388608.0
    elif 폭 == 4:
        자료 = np.frombuffer(원본, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 샘플 폭이다: {폭}")

    if 채널 > 1:
        자료 = 자료.reshape(-1, 채널).mean(axis=1)

    return np.clip(자료, -1.0, 1.0).astype(np.float32), 표본율


def resample(samples: np.ndarray, rate: int, new_rate: int) -> np.ndarray:
    """선형 보간으로 표본율을 바꾼다. 마이크 자리의 소리는 16 · 44.1 kHz 에서 48 kHz 로
    올리는 쪽이라 접힘이 생기지 않는다."""
    samples = np.asarray(samples, dtype=np.float32)
    if rate == new_rate or len(samples) == 0:
        return samples
    n = int(round(len(samples) * new_rate / rate))
    자리 = np.arange(n) * (rate / new_rate)
    return np.interp(자리, np.arange(len(samples)), samples).astype(np.float32)


def to_mic(block: np.ndarray) -> np.ndarray:
    """−1 ~ 1 float 소리를 마이크 블록 형(int16)으로 바꾼다 (계약 2-3)."""
    return np.clip(np.round(np.asarray(block, dtype=np.float32) * 32767.0), -32768, 32767).astype(np.int16)


class WavFile(AudioSource):
    """들고 있는 소리를 블록으로 잘라 낸다. 다 나가면 `None` 이다.

    표본율이 마이크와 다르면 만들 때 마이크 표본율로 바꿔 든다.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self.samples = resample(samples, int(sample_rate), 기본_표본율)
        self.sample_rate = 기본_표본율
        self._커서 = 0

    @classmethod
    def from_path(cls, path: Path | str) -> WavFile:
        자료, 표본율 = load_wav(path)
        return cls(자료, 표본율)

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate

    def block(self, n: int) -> np.ndarray | None:
        if n <= 0 or self._커서 >= len(self.samples):
            return None
        끝 = min(self._커서 + n, len(self.samples))
        블록 = self.samples[self._커서 : 끝]
        self._커서 = 끝
        return 블록

    def rewind(self) -> None:
        """처음부터 다시 내보낸다."""
        self._커서 = 0


def tts(
    text: str,
    voice: str = 기본_TTS_음성,
    sample_rate: int = 기본_표본율,
) -> WavFile:
    """글을 macOS 내장 음성으로 읽어 `WavFile` 로 낸다.

    **외부 서비스를 쓰지 아니한다.** `say` 는 macOS 에 들어 있어 망도 키도
    필요 없고, 3 초 분량을 0.6 초에 만든다. 음성 목록은 `say -v '?'` 로 본다.
    """
    if not text.strip():
        raise ValueError("읽을 글이 비었다")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        경로 = Path(f.name)
    try:
        subprocess.run(
            [
                "say", "-v", voice,
                "-o", str(경로),
                f"--data-format=LEI16@{int(sample_rate)}",
                text,
            ],
            check=True,
            capture_output=True,
        )
        return WavFile.from_path(경로)
    except FileNotFoundError:
        raise RuntimeError("`say` 가 없다. macOS 밖에서는 TTS 를 쓸 수 없다") from None
    except subprocess.CalledProcessError as 오류:
        raise RuntimeError(
            f"`say` 가 실패했다 (음성 {voice!r}): {오류.stderr.decode(errors='replace').strip()}"
        ) from None
    finally:
        경로.unlink(missing_ok=True)


def tts_voices(language: str = "ko_KR") -> tuple[str, ...]:
    """쓸 수 있는 macOS 음성 이름을 낸다. 못 물어보면 기본 음성 하나다.

    맥이 아닌 곳에서도 앱은 서야 하므로 실패를 예외로 올리지 아니한다.
    """
    try:
        결과 = subprocess.run(
            ["say", "-v", "?"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return (기본_TTS_음성,)

    음성들 = []
    for 줄 in 결과.stdout.splitlines():
        if language not in 줄:
            continue
        # `Yuna                ko_KR    # 안녕하세요...` 에서 앞의 이름만 뗀다.
        이름 = 줄.split(language)[0].strip()
        # 한국어 음성 이름에 `Eddy (한국어(한국))` 처럼 괄호가 붙는 것이 있다.
        이름 = 이름.split("(")[0].strip()
        if 이름 and 이름 not in 음성들:
            음성들.append(이름)

    if 기본_TTS_음성 in 음성들:
        음성들.remove(기본_TTS_음성)
        음성들.insert(0, 기본_TTS_음성)
    return tuple(음성들) or (기본_TTS_음성,)


def input_devices() -> tuple[tuple[int, str], ...]:
    """소리를 받을 수 있는 장치를 (번호, 이름) 으로 낸다.

    **기본 장치를 그냥 쓰지 아니한다.** macOS 의 기본 입력은 블루투스 이어폰이
    붙어 있으면 그 쪽이라, 맥북 마이크를 쓰려면 골라 줘야 한다. 못 물어보면
    빈 목록이고, 그때는 앱이 기본 장치를 쓴다.
    """
    try:
        import sounddevice as sd

        return tuple(
            (번호, 장치["name"])
            for 번호, 장치 in enumerate(sd.query_devices())
            if 장치["max_input_channels"] > 0
        )
    except Exception:
        return ()


class Mic(AudioSource):
    """맥북 마이크. 딴 실에서 받아 쌓고 물으면 한 주기 분량을 잘라 낸다."""

    def __init__(self, sample_rate: int = 기본_표본율, device=None) -> None:
        import sounddevice as sd

        self.sample_rate = int(sample_rate)
        self._상한 = int(self.sample_rate * 마이크_보관_s)
        self._버퍼 = np.zeros(0, dtype=np.float32)
        self._자물쇠 = threading.Lock()

        try:
            self._스트림 = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._받는다,
                device=device,
            )
            self._스트림.start()
        except Exception as 오류:
            raise RuntimeError(
                f"마이크를 열지 못했다: {오류}. "
                "시스템 설정 > 개인정보 보호 및 보안 > 마이크 를 확인한다"
            ) from None

    def _받는다(self, indata, frames, time_info, status) -> None:
        with self._자물쇠:
            self._버퍼 = np.concatenate([self._버퍼, indata[:, 0].copy()])
            # 앱 고리가 서 있는 동안에도 마이크는 계속 들어온다. 상한을 안 두면
            # 고리가 다시 돌 때 한참 전 소리부터 나간다.
            if len(self._버퍼) > self._상한:
                self._버퍼 = self._버퍼[-self._상한 :]

    def block(self, n: int) -> np.ndarray | None:
        if n <= 0:
            return None
        with self._자물쇠:
            if len(self._버퍼) < n:
                return None
            블록, self._버퍼 = self._버퍼[:n], self._버퍼[n:]
        return 블록

    def close(self) -> None:
        self._스트림.stop()
        self._스트림.close()
