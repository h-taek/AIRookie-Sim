"""입력원 시험.

프레임 시퀀스의 시간 배분, 레터박스, wav 블록 분할, 발화 목록 읽기를 본다.
사양은 `docs/spec/01_simulator.md` 3장이다.

**기본 시험은 마이크·카메라·`say` 를 건드리지 아니한다.** 실장치가 필요한
것은 `device` 로 표시해 두고 따로 돌린다 -- 장치가 없는 기계나 CI 에서
기본 시험이 빨갛게 되면 안 된다.

    uv run pytest -m "not device"    기본 (기본값이기도 하다)
    uv run pytest -m device          실장치
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from sources import (
    VideoClip,
    VideoFile,
    WavFile,
    fit_frame,
    load_wav,
    video_clips,
)


def 가짜_프레임들(개수: int, 값_시작: int = 0) -> list[np.ndarray]:
    """칸마다 값이 다른 프레임이다. 어느 칸이 나왔는지 값으로 가린다."""
    return [
        np.full((112, 112, 3), 값_시작 + i, dtype=np.uint8) for i in range(개수)
    ]


# --- 그림 맞추기 ---


def test_정사각_얼굴을_2대1_화면에_레터박스로_넣는다():
    """비율을 바꾸지 아니한다. 남는 자리는 검정으로 둔다.

    늘려서 채우면 얼굴이 가로로 퍼져 표정이 달라 보인다. 감정을 읽는 것이
    목적이므로 늘리지 아니한다.
    """
    얼굴 = np.full((112, 112, 3), 200, dtype=np.uint8)
    결과 = fit_frame(얼굴, (960, 480))

    assert 결과.shape == (480, 960, 3)
    assert 결과.dtype == np.uint8
    # 세로를 꽉 채우므로 가로는 480 이 되고 좌우에 240 씩 남는다.
    assert np.all(결과[:, :240] == 0)
    assert np.all(결과[:, 720:] == 0)
    assert np.all(결과[:, 240:720] == 200)


def test_크기가_같으면_그대로_낸다():
    프레임 = np.full((480, 960, 3), 7, dtype=np.uint8)
    결과 = fit_frame(프레임, (960, 480))
    np.testing.assert_array_equal(결과, 프레임)


# --- wav ---


def test_wav를_블록으로_나눠_내고_다_나가면_None이다():
    소리 = np.linspace(-1.0, 1.0, 2500, dtype=np.float32)
    원본 = WavFile(소리, sample_rate=48000)

    블록1 = 원본.block(1000)
    assert 블록1 is not None and len(블록1) == 1000
    np.testing.assert_allclose(블록1, 소리[:1000])

    블록2 = 원본.block(1000)
    assert 블록2 is not None and len(블록2) == 1000

    블록3 = 원본.block(1000)
    assert 블록3 is not None and len(블록3) == 500

    assert 원본.block(1000) is None


def test_되감으면_처음부터_다시_나간다():
    소리 = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    원본 = WavFile(소리, sample_rate=48000)
    원본.block(1000)
    assert 원본.block(10) is None

    원본.rewind()
    다시 = 원본.block(10)
    assert 다시 is not None
    np.testing.assert_allclose(다시, 소리[:10])


def test_길이를_초로_낸다():
    원본 = WavFile(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert 원본.duration_s == pytest.approx(1.0)


def test_wav를_읽어_모노_float32로_낸다(tmp_path):
    모노 = tmp_path / "mono.wav"
    with wave.open(str(모노), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.array([-32768, 0, 16384, 32767], dtype=np.int16).tobytes())

    오디오, sr = load_wav(모노)
    assert sr == 16000
    assert 오디오.dtype == np.float32
    np.testing.assert_allclose(오디오, [-1.0, 0.0, 0.5, 32767 / 32768.0], atol=1e-4)


def test_스테레오_wav는_채널을_평균낸다(tmp_path):
    스테레오 = tmp_path / "stereo.wav"
    with wave.open(str(스테레오), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(
            np.array([16384, -16384, -16384, 16384], dtype=np.int16).tobytes()
        )

    오디오, sr = load_wav(스테레오)
    assert sr == 44100
    assert len(오디오) == 2
    np.testing.assert_allclose(오디오, [0.0, 0.0], atol=1e-4)


def test_wav_파일에서_바로_만든다(tmp_path):
    경로 = tmp_path / "a.wav"
    with wave.open(str(경로), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())

    원본 = WavFile.from_path(경로)
    # 마이크 표본율로 바꿔 든다. 길이는 그대로다.
    assert 원본.sample_rate == 48000
    assert 원본.duration_s == pytest.approx(0.5)


def test_다른_표본율의_소리는_마이크_표본율로_바꿔_든다():
    소리 = np.sin(2 * np.pi * 100 * np.arange(16000) / 16000).astype(np.float32)
    원본 = WavFile(소리, sample_rate=16000)
    assert len(원본.samples) == 48000
    # 같은 시각의 값이 같다.
    np.testing.assert_allclose(원본.samples[::3][:1000], 소리[:1000], atol=1e-6)


# --- 발화 목록 ---


def 가짜_mp4(경로: Path, 너비: int = 64, 높이: int = 32, fps: float = 10.0, 칸수: int = 12) -> Path:
    cv2 = pytest.importorskip("cv2")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(경로), fourcc, fps, (너비, 높이))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter 로 mp4 를 만들지 못했다")
    for i in range(칸수):
        # 0번째 칸은 BGR 순빨강(B=0, G=0, R=255)이고 다른 칸은 다른 색이다.
        프레임 = np.zeros((높이, 너비, 3), dtype=np.uint8)
        if i == 0:
            프레임[:, :, 2] = 255
        else:
            프레임[:, :, 0] = (i * 20) % 256
            프레임[:, :, 1] = (i * 10) % 256
        writer.write(프레임)
    writer.release()
    return 경로


def test_영상_반복이면_끝_뒤에_앞_칸으로_되감는다(tmp_path):
    mp4 = 가짜_mp4(tmp_path / "clip.mp4", 칸수=12, fps=10.0)
    vf = VideoFile.from_path(mp4, loop=True)

    첫_칸 = vf.frame(0.0)
    assert 첫_칸 is not None
    # 1.2 초 뒤(한 바퀴가 돈 뒤)에 0 번째 칸으로 되감긴다.
    끝_뒤_칸 = vf.frame(1.2)
    assert 끝_뒤_칸 is not None
    np.testing.assert_array_equal(끝_뒤_칸, 첫_칸)


def test_영상_반복이_아니면_끝_뒤_None이다(tmp_path):
    mp4 = 가짜_mp4(tmp_path / "clip.mp4", 칸수=12, fps=10.0)
    vf = VideoFile.from_path(mp4, loop=False)

    assert vf.frame(0.0) is not None
    assert vf.frame(1.2) is None
    assert vf.frame(2.0) is None


def test_영상_같은_칸을_두_번_물으면_두_번째는_None이다(tmp_path):
    """직전 것을 다시 싣지 아니한다."""
    mp4 = 가짜_mp4(tmp_path / "clip.mp4", 칸수=12, fps=10.0)
    vf = VideoFile.from_path(mp4, loop=True)

    assert vf.frame(0.0) is not None
    assert vf.frame(0.04) is None
    assert vf.frame(0.10) is not None


def test_영상_나오는_그림이_RGB다(tmp_path):
    """BGR 로 쓴 순빨강이 R 채널로 나온다."""
    mp4 = 가짜_mp4(tmp_path / "clip.mp4", 칸수=12, fps=10.0)
    vf = VideoFile.from_path(mp4, loop=True)

    프레임 = vf.frame(0.0)
    assert 프레임 is not None
    # RGB 에서 R 채널이 지배적이다 (BGR 순빨강이 R 채널로 나온다).
    assert 프레임[0, 0, 0] > 200
    assert 프레임[0, 0, 1] < 30
    assert 프레임[0, 0, 2] < 30


def test_영상_size로_맞춰진다(tmp_path):
    mp4 = 가짜_mp4(tmp_path / "clip.mp4", 너비=64, 높이=32)
    vf = VideoFile.from_path(mp4, size=(960, 480))

    프레임 = vf.frame(0.0)
    assert 프레임 is not None
    assert 프레임.shape == (480, 960, 3)


def test_video_clips가_index_csv를_읽는다(tmp_path):
    (tmp_path / "index.csv").write_text(
        "clip_id,label,text,video,wav,source,start_s,end_s\n"
        "keti_01,neutral,예 쏘겠습니다,keti_01.mp4,keti_01.wav,KETI_01,27.1,29.6\n",
        encoding="utf-8",
    )

    목록 = video_clips(tmp_path)
    assert len(목록) == 1
    assert 목록[0].clip_id == "keti_01"
    assert 목록[0].label == "neutral"
    assert 목록[0].text == "예 쏘겠습니다"
    assert 목록[0].video == tmp_path / "keti_01.mp4"
    assert 목록[0].wav == tmp_path / "keti_01.wav"
    assert 목록[0].source == "KETI_01"
    assert 목록[0].start_s == 27.1
    assert 목록[0].end_s == 29.6


def test_video_clips_자료가_없으면_빈_목록이다(tmp_path):
    assert video_clips(tmp_path / "없다") == []


# --- 실장치 ---


@pytest.mark.device
def test_TTS가_읽은_소리를_바로_쓴다():
    """macOS `say` 로 만든 소리가 그대로 마이크 블록이 된다."""
    from sources import tts

    소리 = tts("루키야 이리 와 봐")
    assert 소리.sample_rate == 48000
    assert 소리.duration_s > 0.3
    블록 = 소리.block(320)
    assert 블록 is not None and len(블록) == 320
    assert np.abs(소리.samples).max() > 0.01


def 기다린다(내놓는다, 제한_s: float = 5.0):
    """장치가 첫 것을 내놓을 때까지 기다린다. 못 내놓으면 None 이다.

    **정해진 시간을 자고 한 번만 묻지 아니한다.** 장치가 깨어나는 데 드는
    시간이 그때그때 달라서, 짧게 자면 조용한 기계에서만 빨갛게 되는 시험이
    된다 -- 무엇이 고장인지 가릴 수 없는 시험이다.
    """
    import time

    마감 = time.monotonic() + 제한_s
    while time.monotonic() < 마감:
        것 = 내놓는다()
        if 것 is not None:
            return 것
        time.sleep(0.05)
    return None


@pytest.mark.device
def test_맥북_마이크에서_블록이_나온다():
    """**기본 장치를 쓰지 아니한다.** 이어폰이 붙어 있으면 그 쪽이 기본이다."""
    from sources import Mic, input_devices

    맥북 = next((번호 for 번호, 이름 in input_devices() if "MacBook" in 이름), None)
    if 맥북 is None:
        pytest.skip("맥북 마이크가 없다")

    마이크 = Mic(device=맥북)
    try:
        블록 = 기다린다(lambda: 마이크.block(320))
        assert 블록 is not None
        assert len(블록) == 320
        assert 블록.dtype == np.float32
    finally:
        마이크.close()


@pytest.mark.device
def test_맥북_카메라에서_프레임이_나온다():
    from sources import Webcam

    캠 = Webcam(size=(960, 480))
    try:
        프레임 = 기다린다(lambda: 캠.frame(0.0))
        assert 프레임 is not None
        assert 프레임.shape == (480, 960, 3)
        # 비율을 지키므로 좌우에 검은 띠가 남는다. 16:9 를 2:1 에 넣으면
        # 한 쪽에 53 픽셀씩이다.
        assert 프레임[:, :40].max() == 0
        assert 프레임[:, -40:].max() == 0
    finally:
        캠.close()


@pytest.mark.device

def test_영상_되감으면_처음부터_다시_나온다():
    영상 = VideoFile(가짜_프레임들(4), fps=4.0, loop=False)
    assert 영상.frame(0.0) is not None
    assert 영상.frame(0.0) is None

    영상.rewind()
    assert 영상.frame(0.0)[0, 0, 0] == 0


def test_영상_프레임이_없으면_만들_때_막는다():
    with pytest.raises(ValueError):
        VideoFile([], fps=30.0)
