#!/usr/bin/env python3
"""Repository path helpers (leaf module; pathlib only)."""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
