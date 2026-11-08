"""외부 주입 트리거.

시뮬레이터가 구조적으로 만들 수 없는 입력을 외부에서 넣는다.
사양은 `docs/spec/01_simulator.md` 3장이 정본이다.

**판정 결과를 넣지 아니한다.** 넣는 것은 호출어가 든 소리이고 얼굴이 담긴
그림이다. 그것이 호출인지, 무슨 감정인지는 온보드 SW 가 정한다.

검증 모드 둘을 여기서 가른다 (사양 3-3).

| 모드 | 로봇 카메라 프레임 |
|---|---|
| 추종 | 월드 렌더. 3D 사람을 따라가는 것을 본다 |
| 감정 표현 | `sources` 의 프레임. 표정을 읽는 것을 본다 |

**소리는 모드를 가리지 아니한다.** 추종 모드에서도 호출어를 넣어야 하므로
(듣고 돌아보는 것이 추종의 시작이다) 소리는 두 모드에서 다 흐른다.
"""

from __future__ import annotations

import numpy as np

from sources import AudioSource, FrameSource, to_mic

# 검증 모드 둘이다. 앱의 라디오가 이 값을 쓴다.
모드_추종 = "추종"
모드_감정 = "감정 표현"
모드들 = (모드_추종, 모드_감정)


class Injector:
    """외부 주입 트리거. 앱의 좌측 패널이 부른다."""

    def __init__(self, bridge) -> None:
        self._bridge = bridge

        self.mode = 모드_추종
        self._frames: FrameSource | None = None
        self._audio: AudioSource | None = None
        # 재생을 시작한 시뮬 시각이다. 프레임 시퀀스는 발화가 시작된 뒤로
        # 얼마가 지났는지로 칸을 고르므로 기준이 있어야 한다.
        self._시작_t = 0.0

    # 1~2. 발화 재생과 얼굴 재생

    def set_mode(self, mode: str) -> None:
        """검증 모드를 바꾼다."""
        if mode not in 모드들:
            raise ValueError(f"모르는 모드다: {mode!r}")
        self.mode = mode

    def play(
        self,
        t: float,
        frames: FrameSource | None = None,
        audio: AudioSource | None = None,
    ) -> None:
        """입력원을 장전하고 `t` 부터 재생한다.

        앞에 틀던 것은 닫고 버린다. 카메라와 마이크는 열어 둔 채로 두면 장치를
        붙잡고 있으므로 바꿀 때 반드시 닫아야 한다.
        """
        if frames is not None and frames is not self._frames:
            self._닫는다(self._frames)
            self._frames = frames
        if audio is not None and audio is not self._audio:
            self._닫는다(self._audio)
            self._audio = audio
        self._시작_t = float(t)

    def stop(self) -> None:
        """틀던 것을 멈추고 장치를 놓는다."""
        self._닫는다(self._frames)
        self._닫는다(self._audio)
        self._frames = None
        self._audio = None

    @staticmethod
    def _닫는다(원본) -> None:
        if 원본 is not None:
            원본.close()

    def frame(self, t: float) -> np.ndarray | None:
        """로봇 카메라 자리에 넣을 프레임을 낸다.

        추종 모드에서는 아무것도 내지 아니한다 -- 그 모드의 프레임은 월드
        렌더이며 여기가 낼 것이 아니다.
        """
        if self.mode != 모드_감정 or self._frames is None:
            return None
        return self._frames.frame(t - self._시작_t)

    def audio_block(self, dt: float) -> np.ndarray | None:
        """한 주기(dt 초) 분량을 마이크 블록으로 잘라 낸다.

        **표본 수를 고정값으로 자르지 아니한다.** 소스의 표본율로 자른다. 블록은
        마이크 형(int16)으로 낸다 (계약 2-3).
        """
        if self._audio is None:
            return None
        블록 = self._audio.block(int(self._audio.sample_rate * dt))
        return None if 블록 is None else to_mic(블록)

    @property
    def playing(self) -> bool:
        """틀고 있는 것이 있는가. 앱이 글자로 띄운다."""
        return self._frames is not None or self._audio is not None

    # 3~4. 전압 강하 · 온도 상승

    def drop_voltage(self, volt: float) -> None:
        """서보가 보고할 전압을 바꾼다. 값을 따로 들고 있지 아니한다."""
        self._bridge.set_voltage(volt)

    def raise_temperature(self, celsius: float) -> None:
        """서보가 보고할 온도를 바꾼다. 값을 따로 들고 있지 아니한다."""
        self._bridge.set_temperature(celsius)

    def clear(self) -> None:
        """전압은 계산값으로, 온도는 기본값으로 되돌린다."""
        self._bridge.set_voltage(None)
        self._bridge.set_temperature(self._bridge.기본_온도_C)
