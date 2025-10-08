"""Run hashing utilities."""
from __future__ import annotations

import hashlib
import json
from typing import Dict

from . import config as cfg_mod


def run_hash(cfg: Dict, versions: Dict) -> str:
    """Generate a deterministic hash for a configuration and version info."""
    canonical_cfg = cfg_mod.canonicalize(cfg)
    payload = {
        "config": canonical_cfg,
        "versions": versions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
