#!/usr/bin/env python3
"""Leow et al. comparison across the three asset sets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from benchmark.utils.benchmark_utils import (
    ROOT, BENCH, CFG, CONFIGS, _load_optimizer_config,
    _print_forecast, _print_portfolio, _dry, _run_named_literature,
)


def task_leow(smoke: bool = False, dry: bool = False):
    for x in ("leow-allweather",):
        _run_named_literature(x, smoke, dry)

