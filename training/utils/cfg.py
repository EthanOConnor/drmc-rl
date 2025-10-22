from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


@dataclass(slots=True)
class ConfigNode:
    """Thin attribute-access wrapper around nested dictionaries."""

    data: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        if item in self.data:
            return self.data[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "data":
            object.__setattr__(self, key, value)
        else:
            self.data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {k: _to_plain(v) for k, v in self.data.items()}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ConfigNode({json.dumps(self.to_dict(), indent=2, sort_keys=True)})"


def _to_plain(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _merge_dicts(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _merge_dicts(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping, received {type(data)!r}")
    return data


def load_and_merge_cfg(base_path: str | Path, *extra_paths: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    for path in extra_paths:
        if path is None:
            continue
        cfg = _merge_dicts(cfg, load_yaml(path))
    return cfg


def apply_dot_overrides(cfg: MutableMapping[str, Any], overrides: str | None) -> None:
    if not overrides:
        return
    for item in overrides.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], MutableMapping):
                target[part] = {}
            target = target[part]  # type: ignore[assignment]
        target[parts[-1]] = _parse_override_value(raw_value)


def _parse_override_value(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if token.startswith("0x"):
            return int(token, 16)
        if token.startswith("0b"):
            return int(token, 2)
        if token.startswith("0o"):
            return int(token, 8)
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)
    except ValueError:
        return token


def to_config_node(data: Mapping[str, Any]) -> ConfigNode:
    return ConfigNode({k: _wrap(v) for k, v in data.items()})


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value
