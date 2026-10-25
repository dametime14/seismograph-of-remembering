"""설정 파일 읽기.

YAML 을 읽고 ${a.b} 형태를 같은 파일의 키나 환경변수로 치환한다.
명령행에서 --set chain.n_iters=5 처럼 값을 덮어쓸 수 있다.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_PATTERN = re.compile(r"\$\{([^}]+)\}")


class Config(dict):
    """점 표기 접근이 가능한 dict. cfg.chain.n_iters 처럼 쓴다."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str) -> Path:
        """paths.xxx 같은 키를 저장소 루트 기준의 절대경로로 돌려준다."""
        value = lookup(self, dotted)
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = (Path(self["_root"]) / p).resolve()
        return p


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def lookup(cfg: Dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _assign(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _resolve(cfg: Dict[str, Any], max_rounds: int = 10) -> None:
    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            def repl(match: "re.Match[str]") -> str:
                key = match.group(1)
                if key in os.environ:
                    return os.environ[key]
                try:
                    return str(lookup(cfg, key))
                except KeyError:
                    return match.group(0)
            return _PATTERN.sub(repl, value)
        if isinstance(value, dict):
            for k in list(value.keys()):
                value[k] = resolve_value(value[k])
            return value
        if isinstance(value, list):
            return [resolve_value(v) for v in value]
        return value

    for _ in range(max_rounds):
        before = yaml.safe_dump(cfg, sort_keys=True)
        resolve_value(cfg)
        if yaml.safe_dump(cfg, sort_keys=True) == before:
            break


def load_config(path: str | os.PathLike, overrides: Optional[List[str]] = None) -> Config:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set 은 key=value 형태여야 한다: {item}")
        key, val = item.split("=", 1)
        _assign(raw, key.strip(), yaml.safe_load(val))
    _resolve(raw)
    # 저장소 루트: configs/ 의 부모를 기본으로, paths.root 가 있으면 그 기준
    repo_root = path.parent.parent
    root = Path(str(raw.get("paths", {}).get("root", "."))).expanduser()
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    raw["_root"] = str(root)
    raw["_config_path"] = str(path)
    return _wrap(raw)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/default.yaml", help="설정 YAML 경로")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="설정 덮어쓰기. 예: --set chain.n_iters=5 backend=mock")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.set)
