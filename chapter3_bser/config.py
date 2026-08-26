"""Frozen Phase-1A configuration loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "chapter3" / "bser_phase1a.json"
PHASE1A1_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "chapter3" / "bser_phase1a1.json"


def load_bser_config(path: Optional[Path] = None) -> Dict[str, Any]:
    with (Path(path) if path is not None else DEFAULT_CONFIG).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema") != "bser.phase1a.config.v1" or not config.get("frozen_before_e1"):
        raise ValueError("BSER Phase-1A configuration is not frozen")
    return config


def load_bser_phase1a1_config(path: Optional[Path] = None) -> Dict[str, Any]:
    with (Path(path) if path is not None else PHASE1A1_CONFIG).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema") != "bser.phase1a1.config.v1" or config.get("formal_training") is not False:
        raise ValueError("invalid BSER Phase-1A.1 configuration")
    if config["e1_v2"]["snapshot_steps"] != [0, 10, 25, 50]:
        raise ValueError("E1-v2 snapshot protocol changed")
    return config
