import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dflop_config.yaml"


def _resolve_relative(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def get_config_path() -> Path:
    env_path = os.environ.get("DFLOP_CONFIG")
    if env_path:
        return _resolve_relative(env_path, _DEFAULT_CONFIG_PATH.parent)
    return _DEFAULT_CONFIG_PATH


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    config_path = get_config_path()
    with config_path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {config_path} must be a mapping.")
    return data


def resolve_path(value: Optional[str], base: Optional[Path] = None) -> Optional[Path]:
    if value is None:
        return None
    base_dir = base if base is not None else get_config_path().parent
    return _resolve_relative(value, base_dir)


def reset_config_cache() -> None:
    load_config.cache_clear()
