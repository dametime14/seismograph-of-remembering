"""스크립트에서 seismo 패키지를 설치 없이 불러오기 위한 경로 설정."""
import functools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = functools.partial(print, flush=True)
