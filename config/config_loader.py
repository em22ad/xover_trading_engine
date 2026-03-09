from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    """
    Load config.yaml from the trading_engine package directory
    (the directory that contains main.py).

    Layout:
        trading_engine/
            main.py
            config.yaml
            config/
                config_loader.py
            ...
    """
    # config_loader.py is in trading_engine/config/
    # so parent of parent is trading_engine/
    package_root = Path(__file__).resolve().parents[1]
    cfg_path = package_root / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")

    with cfg_path.open("r") as f:
        return yaml.safe_load(f)
