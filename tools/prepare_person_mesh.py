"""받아 온 사람 3D 모델을 데모가 쓸 수 있는 형태로 맞춘다.

    uv run --extra demo python tools/prepare_person_mesh.py 내려받은것.obj

모델마다 위쪽 축도, 단위도, 면의 꼭짓점 수도 제각각이다. 그대로 넣으면
사람이 누워 있거나, 3 cm 로 서 있거나, MuJoCo 가 읽기를 거부한다. 이
스크립트가 맞춰 주는 것은 넷이다.

- 사각면과 다각면을 삼각형으로 쪼갠다. MuJoCo 는 삼각형만 읽는다
- 위쪽 축을 z 로 돌린다. 사람 모델은 대개 y 가 위다
- 키를 1.7 m 로 맞춘다. 검출 계약의 `h` 가 거리로 읽히려면 실제 사람 키여야 한다
- 원점을 몸의 중심에 둔다. mocap 위치가 사람 박스의 중심이라는 규약이다

그룹 이름으로 일부만 골라 쓸 수도 있다. 사람 모델에는 머리카락·치아·
속옷 같은 보조 형상이 같이 들어 있는 일이 잦다.

    --group body            body 그룹만 쓴다
    --list-groups           어떤 그룹이 있는지만 보고 끝낸다

내는 파일은 저장소에 올리지 않는다. 자산은 각자 로컬에 둔다.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

사람_키_m = 1.7

# obj 의 어느 축이 위인지에 따라 (x, y, z) 를 어떻게 옮길지. 셋 다
# 행렬식이 +1 이라 좌우가 뒤집히지 않는다.
축_옮기기 = {
    "y": lambda V: np.column_stack([V[:, 2], V[:, 0], V[:, 1]]),
    "z": lambda V: V,
    "x": lambda V: np.column_stack([V[:, 1], V[:, 2], V[:, 0]]),
}


def obj_를_읽는다(path: Path) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """정점과 면, 그리고 면마다의 그룹 이름을 낸다."""
    정점: list[tuple[float, float, float]] = []
    면: list[list[int]] = []
    면의_그룹: list[str] = []
    현재 = ""
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            부분 = line.split()
            정점.append((float(부분[1]), float(부분[2]), float(부분[3])))
        elif line.startswith("g ") or line.startswith("o "):
            현재 = line[2:].strip()
        elif line.startswith("f "):
            # "1/2/3" 처럼 텍스처와 법선이 붙어 있어도 정점 번호만 쓴다.
            면.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
            면의_그룹.append(현재)
    if not 정점 or not 면:
        raise SystemExit(f"{path} 에서 정점이나 면을 못 찾았다. OBJ 가 맞는지 본다")
    return np.array(정점, dtype=np.float64), 면, 면의_그룹


def 삼각형으로_쪼갠다(면: list[list[int]]) -> list[tuple[int, int, int]]:
    """다각면을 부채꼴로 쪼갠다. 볼록하지 않으면 조금 뭉개지지만
    사람 모델의 사각면에서는 문제가 되지 않는다."""
    나온_것 = []
    for 꼭짓점 in 면:
        for i in range(1, len(꼭짓점) - 1):
            나온_것.append((꼭짓점[0], 꼭짓점[i], 꼭짓점[i + 1]))
    return 나온_것


def 맞춘다(V: np.ndarray, up: str) -> np.ndarray:
    V = 축_옮기기[up](V)
    높이 = V[:, 2].max() - V[:, 2].min()
    if 높이 <= 0:
        raise SystemExit("높이가 0이다. --up 을 다른 축으로 줘 본다")
    V = V * (사람_키_m / 높이)
    for 축 in range(3):
        V[:, 축] -= (V[:, 축].max() + V[:, 축].min()) / 2.0
    return V


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("입력", help="받아 온 OBJ 파일")
    p.add_argument("-o", "--out", default="assets/person.obj")
    p.add_argument("--up", default="y", choices=sorted(축_옮기기), help="입력 모델에서 위쪽 축")
    p.add_argument("--group", action="append", help="이 그룹만 쓴다. 여러 번 줄 수 있다")
    p.add_argument("--list-groups", action="store_true", help="그룹만 보고 끝낸다")
    args = p.parse_args()

    입력 = Path(args.입력)
    V, 면, 면의_그룹 = obj_를_읽는다(입력)

    if args.list_groups:
        센_것 = Counter(면의_그룹)
        print(f"{입력.name} — 정점 {len(V)}, 면 {len(면)}")
        for 이름, 수 in 센_것.most_common():
            print(f"  {이름 or '(이름 없음)':32s} 면 {수}")
        return

    if args.group:
        고른_것 = [f for f, g in zip(면, 면의_그룹) if g in args.group]
        if not 고른_것:
            raise SystemExit(f"{args.group} 에 해당하는 면이 없다. --list-groups 로 본다")
        면 = 고른_것

    쓰인_것 = sorted({i for f in 면 for i in f})
    새_번호 = {옛: 새 for 새, 옛 in enumerate(쓰인_것)}
    V = 맞춘다(V[쓰인_것], args.up)
    삼각형 = 삼각형으로_쪼갠다([[새_번호[i] for i in f] for f in 면])

    낼_곳 = Path(args.out)
    낼_곳.parent.mkdir(parents=True, exist_ok=True)
    with 낼_곳.open("w") as fp:
        fp.write(f"# {입력.name} 에서 만들었다. 키 {사람_키_m} m, z 가 위, 원점은 몸의 중심.\n")
        for x, y, z in V:
            fp.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for a, b, c in 삼각형:
            fp.write(f"f {a + 1} {b + 1} {c + 1}\n")

    print(f"{낼_곳} — 정점 {len(V)}, 삼각형 {len(삼각형)}, {낼_곳.stat().st_size // 1024} KB")
    print(f"  앞뒤 x [{V[:, 0].min():+.3f}, {V[:, 0].max():+.3f}]")
    print(f"  좌우 y [{V[:, 1].min():+.3f}, {V[:, 1].max():+.3f}]")
    print(f"  높이 z [{V[:, 2].min():+.3f}, {V[:, 2].max():+.3f}]")
    print("  데모에서 앞뒤로 얇고 좌우로 넓게 나오는지 본다. 뒤집혀 있으면 --up 을 바꾼다")


if __name__ == "__main__":
    main()
