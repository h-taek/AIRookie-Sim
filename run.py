#!/usr/bin/env python
"""앱을 띄운다.

경로를 잡아 주는 것이 전부다. 온보드 SW 는 설치하지 않고 경로로 붙이므로
(`docs/README.md`), 맨 파이썬으로는 `src` 두 곳이 보이지 않는다.
"""

import sys
from pathlib import Path

뿌리 = Path(__file__).resolve().parent
sys.path[:0] = [str(뿌리 / "src"), str(뿌리.parent / "AIRookie_Runtime" / "src")]

from app import main

if __name__ == "__main__":
    main()
