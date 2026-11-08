"""감정 경로 대조에 쓸 발화 자료를 학습 서버에서 받아 온다.

앱은 이 자료를 쓰지 않는다. 얼굴 크롭만 든 그림은 사람으로 잘 안 잡혀
`FOLLOW` 에 들지 못한다. Runtime 의 `tools/check_emotion_parity.py` 가 읽는다.

감정 모델팀이 AI Hub "한국인 감정인식을 위한 복합영상"(멀티모달영상,
dataSetSn=58) 을 가공해 둔 것을 쓴다. 발화 하나가 **소리 wav 한 개와 얼굴
프레임 여러 장**으로 되어 있고, 매니페스트에 정답 감정과 전사문이 붙어 있다.

**저장소에 담지 아니한다.** `assets/person.obj` 와 같다 -- 원본이 AI Hub
이용 약관 아래 있고 용량도 크다. 각자 받아 로컬에 둔다.

    uv run --extra demo python tools/fetch_emotion_samples.py

접속 정보는 저장소 루트의 `.env` 에서 읽는다. 스크립트에 박지 아니한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

# 이 저장소 루트와 그 위 작업 폴더다. `.env` 는 여러 저장소가 같이 쓰므로
# 위에 있다.
저장소 = Path(__file__).resolve().parents[1]
작업_폴더 = 저장소.parent

# 서버에서 가공본이 있는 자리다. 감정 모델팀 저장소의 작업 폴더 기준이다.
서버_뿌리 = "work/trimodal_emotion_model"

# 감정 일곱이다. `AIRookie_Emotion/src/datasets/labels.py` 가 정본이다.
감정들 = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def env를_읽는다(경로: Path) -> dict[str, str]:
    """`.env` 를 읽어 사전으로 낸다.

    값 뒤에 주석이 붙어 있으므로(`10022  # 기본 22 가 아니다`) 잘라 낸다.
    """
    if not 경로.exists():
        raise SystemExit(f"접속 정보가 없다: {경로}\n  .env.example 을 복사해 채운다")

    값 = {}
    for 줄 in 경로.read_text(encoding="utf-8").splitlines():
        줄 = 줄.strip()
        if not 줄 or 줄.startswith("#") or "=" not in 줄:
            continue
        이름, _, 나머지 = 줄.partition("=")
        값[이름.strip()] = 나머지.split("#")[0].strip()
    return 값


def 접속을_짠다(env: dict[str, str]) -> tuple[str, list[str]]:
    """`user@host` 와 ssh 옵션을 낸다."""
    없는_것 = [이름 for 이름 in ("SERVER_USER", "SERVER_HOST", "KEY_PATH") if not env.get(이름)]
    if 없는_것:
        raise SystemExit(f".env 에 값이 비었다: {', '.join(없는_것)}")

    키 = Path(os.path.expanduser(env["KEY_PATH"]))
    if not 키.is_absolute():
        키 = (작업_폴더 / 키).resolve()
    if not 키.exists():
        raise SystemExit(f"개인키를 찾을 수 없다: {키}")

    옵션 = ["-i", str(키), "-p", env.get("SERVER_PORT", "22")]
    return f"{env['SERVER_USER']}@{env['SERVER_HOST']}", 옵션


def 고른다(대상: str, 옵션: list[str], 분할: str, 개수: int) -> list[dict[str, str]]:
    """서버 매니페스트에서 감정마다 개수만큼 고른다.

    **네 판정이 다 맞아떨어지는 발화만 고른다.** 매니페스트에는 사람이 영상·
    소리·글을 따로 본 판정과 종합 판정이 같이 있는데, 종합 판정만 보고 고르면
    얼굴은 무표정인데 정답이 '분노' 인 발화가 섞인다. 그런 것을 넣고 시뮬에서
    보면 무엇이 틀렸는지 가릴 수가 없다.
    """
    원격_코드 = f"""
import csv, json
rows = list(csv.DictReader(open("{서버_뿌리}/data/manifests_si/{분할}.csv")))
고른_것 = []
for 감정 in {감정들!r}:
    맞는_것 = [r for r in rows
               if r["label"] == 감정 == r["label_image"] == r["label_sound"]]
    고른_것 += 맞는_것[:{개수}]
print(json.dumps(고른_것, ensure_ascii=False))
"""
    결과 = subprocess.run(
        ["ssh", *옵션, 대상, f"python3 -c {shlex.quote(원격_코드)}"],
        capture_output=True, text=True,
    )
    if 결과.returncode != 0:
        raise SystemExit(f"서버에서 매니페스트를 읽지 못했다:\n{결과.stderr.strip()}")

    import json

    return json.loads(결과.stdout)


def 받는다(대상: str, 옵션: list[str], 줄들: list[dict[str, str]], 받을_곳: Path) -> None:
    """고른 발화의 wav 와 얼굴 프레임 폴더를 받는다.

    **rsync 를 쓰지 아니한다.** macOS 기본 rsync 는 openrsync 라 옵션이 다르고,
    homebrew rsync 가 깔린 기계와 안 깔린 기계에서 다르게 동작한다. 서버에서
    tar 로 묶어 흘려보내고 여기서 푼다 -- 접속도 한 번이다.
    """
    경로들 = []
    for 줄 in 줄들:
        경로들.append(줄["wav_path"])
        경로들.append(줄["face_frames_dir"])

    받을_곳.mkdir(parents=True, exist_ok=True)
    묶기 = f"tar cf - -C {shlex.quote(서버_뿌리)} " + " ".join(
        shlex.quote(경로) for 경로 in 경로들
    )
    보내기 = subprocess.Popen(
        ["ssh", *옵션, 대상, 묶기], stdout=subprocess.PIPE
    )
    풀기 = subprocess.run(["tar", "xf", "-", "-C", str(받을_곳)], stdin=보내기.stdout)
    보내기.wait()
    if 보내기.returncode != 0 or 풀기.returncode != 0:
        raise SystemExit("자료를 받지 못했다")


def 목록을_쓴다(줄들: list[dict[str, str]], 받을_곳: Path) -> Path:
    """앱이 읽을 목록을 낸다. 경로는 이 폴더 기준 상대 경로다."""
    목록 = 받을_곳 / "index.csv"
    with 목록.open("w", encoding="utf-8", newline="") as f:
        쓰기 = csv.writer(f)
        쓰기.writerow(["utt_id", "label", "text", "wav", "frames"])
        for 줄 in 줄들:
            쓰기.writerow([
                줄["utt_id"], 줄["label"], 줄["text"],
                줄["wav_path"], 줄["face_frames_dir"],
            ])
    return 목록


def main() -> int:
    파서 = argparse.ArgumentParser(description=__doc__)
    파서.add_argument("--count", type=int, default=3, help="감정마다 몇 건 (기본 3)")
    파서.add_argument("--split", default="test", help="매니페스트 분할 (기본 test)")
    파서.add_argument(
        "--out", type=Path, default=저장소 / "assets" / "emotion",
        help="받을 곳 (기본 assets/emotion)",
    )
    인자 = 파서.parse_args()

    env = env를_읽는다(작업_폴더 / ".env")
    대상, 옵션 = 접속을_짠다(env)

    print(f"매니페스트({인자.split})에서 감정마다 {인자.count} 건을 고른다...")
    줄들 = 고른다(대상, 옵션, 인자.split, 인자.count)
    if not 줄들:
        raise SystemExit("고른 것이 없다")

    셈 = {}
    for 줄 in 줄들:
        셈[줄["label"]] = 셈.get(줄["label"], 0) + 1
    print("  " + "  ".join(f"{감정} {셈.get(감정, 0)}" for 감정 in 감정들))

    print(f"{len(줄들)} 건을 받는다 -> {인자.out}")
    받는다(대상, 옵션, 줄들, 인자.out)
    목록 = 목록을_쓴다(줄들, 인자.out)
    print(f"목록을 썼다: {목록}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
