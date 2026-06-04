import pathlib

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml from *path* and return as a dict (empty dict if missing)."""
    p = pathlib.Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}
