"""
src/config.py
--------------
Tiny YAML config loader that returns a dict-like object supporting both
`cfg["sensor"]["mode"]` and `cfg.sensor.mode` access styles, so downstream
modules can be written in whichever style is more readable.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


class AttrDict(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str = "configs/config.yaml") -> AttrDict:
    """Load the YAML config file and return it as an AttrDict."""
    if not os.path.isabs(path):
        # Resolve relative to the repository root (parent of this file's dir).
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(here, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)
    return _wrap(raw)
