"""호출부터 추종까지를 창 없이 벽시계로 돌려 잰다 (Runtime 계획 5-3 통과 판정).

앱과 같은 고리(벽시계 20 ms, 늦으면 세상이 흐른다)에 실제 검출기와 whisper 를
물린다. TTS 로 호출어를 넣고, 사람을 대본대로 움직이며, 모드 전이와 추종 수치를
낸다.

    .venv/bin/python tools/run_scenario.py front     # 오른쪽 앞의 사람을 불러 따라가게 한다
    .venv/bin/python tools/run_scenario.py behind    # 뒤에 있는 사람을 찾게 한다
    .venv/bin/python tools/run_scenario.py emotion   # 다섯 영상 클립으로 감정을 잰다
    .venv/bin/python tools/run_scenario.py emotion --clip keti_0000000086
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

뿌리 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(뿌리 / "src"), str(뿌리.parent / "AIRookie_Runtime" / "src")]

import numpy as np  # noqa: E402

import sources  # noqa: E402
from app import Sim, 감정_영상_폴더, 모드_감정  # noqa: E402

장면들 = {
    # (사람 시작 x, y, 호출 시각, 구간, 사람이 움직이는 대본)
    "front": dict(x=6.0, y=-1.5, 부름=1.0, 구간=60.0,
                  대본=[(35.0, 50.0, (0.0, 0.25))]),  # 35 ~ 50 초에 로봇 왼쪽으로 0.25 m/s 걷는다
    "behind": dict(x=-5.0, y=0.5, 부름=1.0, 구간=60.0, 대본=[]),
}


def 감정_장면(policy: Path | str | None, clip_id: str | None) -> int:
    """영상 자료로 감정 추론 과정을 창 없이 벽시계로 잰다."""
    전체_클립들 = sources.video_clips(감정_영상_폴더)
    if not 전체_클립들:
        print(f"영상 자료가 없다: {감정_영상_폴더}")
        return 1

    if clip_id is not None:
        대상_클립들 = [c for c in 전체_클립들 if c.clip_id == clip_id]
        if not 대상_클립들:
            print(f"클립을 찾을 수 없다: {clip_id}")
            return 1
    else:
        대상_클립들 = 전체_클립들

    호출 = sources.tts("로키야 이리 와")
    요약_표 = []

    for 클립 in 대상_클립들:
        s = Sim() if policy is None else Sim(policy=policy)
        try:
            영상 = sources.VideoFile.from_path(
                클립.video, size=s.views.robot_size, loop=True
            )
            소리 = sources.WavFile.from_path(클립.wav)

            # 1. 감정 표현 모드로 바꾸고 그 클립 영상을 반복으로 흘린다.
            s.set_mode(모드_감정)
            s.injector.play(t=s.t, frames=영상)

            모드들: list[tuple[float, str]] = []
            앞_모드 = None
            표현들: list[tuple[float, str]] = []
            앞_표현 = None
            표현_이름들: list[str] = []
            표현_최저_몸통: float = 1.0
            표현_있었음: bool = False
            호출_틈 = False
            follow_t: float | None = None
            클립_소리_시작_t: float | None = None
            클립_소리_끝_t: float | None = None
            감정_시각: float | None = None
            감정_값: str = "없음"
            감정_note: str = ""
            발화_end_t: float | None = None
            발화_text_t: float | None = None
            낮은 = 1.0

            시작 = time.perf_counter()
            끝남 = False

            while not 끝남:
                경과 = time.perf_counter() - 시작
                n = s.due(경과)
                if not n:
                    time.sleep(0.001)
                    continue

                # 3. t = 1.0 초에 TTS "로키야 이리 와" 를 소리로 넣는다.
                if not 호출_틈 and s.t >= 1.0:
                    s.injector.play(
                        t=s.t,
                        audio=sources.WavFile(호출.samples.copy(), 호출.sample_rate),
                    )
                    호출_틈 = True

                s.cycle(n)
                조회 = s.runtime.snapshot()
                모드 = 조회.judgment.mode

                if 모드 != 앞_모드:
                    모드들.append((round(s.t, 2), 모드))
                    앞_모드 = 모드
                    if 모드 == "FOLLOW" and follow_t is None:
                        follow_t = s.t

                # 표현 상태 변화를 기록한다 (계획 5장 Task 7).
                표현 = 조회.approach.expression
                if 표현 != 앞_표현:
                    표현들.append((round(s.t, 2), 표현))
                    앞_표현 = 표현
                    조각들 = 표현.split()
                    if len(조각들) >= 2:
                        상태, 이름 = 조각들[0], 조각들[1]
                        if 상태 != "OFF" and 이름 != "없음":
                            if not 표현_이름들 or 표현_이름들[-1] != 이름:
                                표현_이름들.append(이름)

                if 조회.approach.expression and not 조회.approach.expression.startswith("OFF"):
                    표현_있었음 = True
                    표현_최저_몸통 = min(표현_최저_몸통, s.world.torso_height())

                낮은 = min(낮은, s.world.torso_height())

                # 4. 종합 판단 모드가 FOLLOW 가 되고 1.0 초 뒤 영상을 되감고 클립 wav 를 같은 시각에 튼다.
                if follow_t is not None and 클립_소리_시작_t is None:
                    if s.t >= follow_t + 1.0:
                        영상.rewind()
                        소리.rewind()
                        s.injector.play(t=s.t, frames=영상, audio=소리)
                        클립_소리_시작_t = s.t
                        클립_소리_끝_t = 클립_소리_시작_t + 소리.duration_s

                # 5. 조회의 judgment.emotion 이 "없다" 에서 바뀌면 그 시각과 값을 적는다.
                #    반응 표현이 끝까지 재생될 수 있도록 감정 결과 도달 후 4.0초간 더 유지한다 (Task 7).
                #    클립 소리 시작 후 25 초 안에 안 바뀌면 "없음" 과 그때의 emotion_note 를 적는다.
                if 클립_소리_시작_t is not None:
                    if 조회.judgment.emotion != "없다":
                        if 감정_시각 is None:
                            감정_시각 = s.t
                            감정_값 = 조회.judgment.emotion
                            발화 = s.senses.perception.utterance
                            if 발화 is not None:
                                발화_end_t = 발화.end_t
                                발화_text_t = 발화.text_t
                        if s.t - 감정_시각 >= 4.0:
                            끝남 = True
                    elif s.t - 클립_소리_시작_t >= 25.0:
                        감정_값 = "없음"
                        감정_note = 조회.judgment.emotion_note
                        발화 = s.senses.perception.utterance
                        if 발화 is not None:
                            발화_end_t = 발화.end_t
                            발화_text_t = 발화.text_t
                        끝남 = True
                else:
                    if s.t >= 35.0:
                        감정_값 = "없음"
                        감정_note = "FOLLOW 미진입 타임아웃"
                        끝남 = True

            # 클립마다 낼 것 출력
            print(f"클립: {클립.clip_id} (라벨: {클립.label})")
            for t_m, m in 모드들:
                print(f"  {t_m:6.2f} s  {m}")
            if follow_t is not None:
                print(f"  FOLLOW 에 든 시각: {follow_t:6.2f} s")
            else:
                print("  FOLLOW 에 든 시각: 없음")

            if 클립_소리_시작_t is not None and 클립_소리_끝_t is not None:
                print(f"  클립 소리 시각: 시작 {클립_소리_시작_t:6.2f} s, 끝 {클립_소리_끝_t:6.2f} s")
            if 발화_end_t is not None and 발화_text_t is not None:
                print(f"  발화 닫힌 시각: end_t {발화_end_t:6.2f} s, text_t {발화_text_t:6.2f} s")

            닫힘_감정_초_표시 = "--"
            if 감정_시각 is not None:
                if 발화_end_t is not None:
                    걸린_초 = 감정_시각 - 발화_end_t
                    닫힘_감정_초_표시 = f"{걸린_초:.2f}"
                    print(f"  감정 칸에 닿은 시각: {감정_시각:6.2f} s (발화 닫힘에서 감정까지 {걸린_초:.2f} 초)")
                else:
                    print(f"  감정 칸에 닿은 시각: {감정_시각:6.2f} s")
                print(f"  감정 값: {감정_값}")
            else:
                print(f"  감정 값: 없음 ({감정_note})")

            # approach.expression 이 바뀐 시각과 값을 목록으로 출력한다.
            if 표현들:
                print("  표현 변화:")
                for t_e, expr in 표현들:
                    print(f"    {t_e:6.2f} s  {expr}")

            print(f"  전사문: {조회.judgment.heard}")
            print(f"  주기 {s.cycles}, 빠진 {s.missed}")
            print(f"  가장 낮은 몸통 {낮은:.3f} m (넘어짐 {'있음' if 낮은 < 0.10 else '없음'})")

            표현_순서 = " → ".join(표현_이름들) if 표현_이름들 else "--"
            표현_최저_몸통_표시 = f"{표현_최저_몸통:.3f} m" if 표현_있었음 else "--"

            # 6. 끝나면 한 줄 요약
            print(
                f"  [한 줄 요약] {클립.clip_id} | {클립.label} | {감정_값} | "
                f"표현: {표현_순서} | 최저(표현): {표현_최저_몸통_표시} | "
                f"닫힘→감정: {닫힘_감정_초_표시} 초 | 빠진 주기: {s.missed}\n"
            )

            요약_표.append((
                클립.clip_id,
                클립.label,
                감정_값,
                표현_순서,
                표현_최저_몸통_표시,
                닫힘_감정_초_표시,
                s.missed,
            ))

        finally:
            s.close()

    # 마지막에 다섯 클립 요약 표 (표현 순서, 표현 중 최저 몸통 열 추가).
    print("=" * 124)
    print(
        f"{'클립':<20} {'라벨':<10} {'감정':<16} "
        f"{'표현 순서':<32} {'표현 중 최저 몸통':<18} "
        f"{'닫힘→감정 초':<12} {'빠진 주기':<10}"
    )
    print("-" * 124)
    for c_id, lbl, emo, expr_seq, min_torso_str, elp, msd in 요약_표:
        print(
            f"{c_id:<20} {lbl:<10} {emo:<16} "
            f"{expr_seq:<32} {min_torso_str:<18} "
            f"{elp:<12} {msd:<10}"
        )
    print("=" * 124)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=list(장면들) + ["emotion"])
    ap.add_argument("--policy", default=None)
    ap.add_argument("--clip", default=None, help="감정 장면에서 특정 클립만 돌린다")
    args = ap.parse_args()

    if args.scene == "emotion":
        return 감정_장면(policy=args.policy, clip_id=args.clip)

    장면 = 장면들[args.scene]

    s = Sim() if args.policy is None else Sim(policy=args.policy)
    s.world.set_person_pos(장면["x"], 장면["y"])
    s.world.set_person_yaw(s.world.로봇을_바라보는_각())
    호출 = sources.tts("로키야 이리 와")

    모드들, cx들, 요들, 낮은 = [], [], [], 1.0
    앞_모드 = None
    튼 = False
    시작 = time.perf_counter()
    앞_벽 = 0.0
    while (경과 := time.perf_counter() - 시작) < 장면["구간"]:
        n = s.due(경과)
        if not n:
            time.sleep(0.001)
            continue
        # 사람을 대본대로 움직인다. 월드 조작은 온보드를 거치지 않는다.
        for 시작_t, 끝_t, (vx, vy) in 장면["대본"]:
            if 시작_t <= s.t < 끝_t:
                x, y = s.world.person_pos
                s.world.set_person_pos(x + vx * (경과 - 앞_벽), y + vy * (경과 - 앞_벽))
                s.world.set_person_yaw(s.world.로봇을_바라보는_각())
        앞_벽 = 경과
        if not 튼 and s.t >= 장면["부름"]:
            s.injector.play(t=s.t, audio=sources.WavFile(호출.samples.copy(), 호출.sample_rate))
            튼 = True

        s.cycle(n)
        조회 = s.runtime.snapshot()
        모드 = 조회.judgment.mode
        if 모드 != 앞_모드:
            발화 = s.senses.perception.utterance
            닫힘 = "" if 발화 is None else f"(발화 {발화.start_t:.2f}~{발화.end_t:.2f} s, 전사 {발화.text_t:.2f} s)"
            모드들.append((round(s.t, 2), 모드, f"{조회.judgment.heard} {닫힘}".strip()))
            앞_모드 = 모드
        if 모드 == "FOLLOW" and 조회.judgment.target != "없다":
            cx = float(조회.judgment.target.split("cx")[1].split()[0])
            cx들.append(cx)
        요들.append(float(조회.approach.command.split()[5]))
        낮은 = min(낮은, s.world.torso_height())

    몸통 = s.world.data.body("base").xpos
    사람 = s.world.person_pos
    거리 = math.hypot(사람[0] - 몸통[0], 사람[1] - 몸통[1])
    부호_전환 = int(np.sum(np.diff(np.sign(np.round(요들, 3))) != 0))
    print(f"장면 {args.scene}, {장면['구간']:.0f} 초, 주기 {s.cycles}, 빠진 {s.missed}")
    for t, 모드, 들은 in 모드들:
        print(f"  {t:6.2f} s  {모드:12s} {('들은 말: ' + 들은) if 들은 else ''}")
    if cx들:
        print(f"  FOLLOW 중 |cx| 평균 {np.mean(np.abs(cx들)):.3f} · 최대 {np.max(np.abs(cx들)):.3f} ({len(cx들)} 주기)")
    print(f"  머리 요 명령 부호 전환 {부호_전환} 회")
    print(f"  가장 낮은 몸통 {낮은:.3f} m (넘어짐 {'있음' if 낮은 < 0.10 else '없음'})")
    print(f"  끝: 로봇–사람 거리 {거리:.2f} m, 몸통 회전 {조회.approach.body_turn_deg} 도, 모드 {앞_모드}")
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
