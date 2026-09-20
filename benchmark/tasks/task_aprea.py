#!/usr/bin/env python3
"""Aprea & Sbaiz (2025) method-substitution benchmark, DJIA and NASDAQ 100.

Their protocol, kept deliberately: monthly decisions over 2016-2020 (60
allocations), training from 2007, w_max = 0.2. K = 10 is *our* addition --
their model has no cardinality constraint, but their Tables 7 and 8 report an
average of 10 non-zero positions at u_i = 0.2, so K = 10 prescribes exactly the
count their interval bound produces on its own.

Both universes are aligned with the sets Aprea & Sbaiz report on (28 DJIA, 54
NASDAQ 100) and are pinned by `expected_assets` + `enforce_expected_assets`, so
a universe that silently changes size aborts the run rather than quietly
comparing a different portfolio against their published Sharpe.

That guard matters most for WBA, which both universes hold: Walgreens went
private and was delisted in 2025, so yfinance now serves no history for it at
all. Each config carries a `series_overrides` entry pointing at a CC0 dataset
for the full 2006-2020 adjusted closes. Dropping the ticker instead would be
survivorship bias -- WBA fell heavily across the out-of-sample window.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.utils.benchmark_utils import _run_named_literature


def task_aprea_djia(smoke: bool = False, dry: bool = False):
    """DJIA universe (n=28)."""
    return _run_named_literature("aprea-djia", smoke, dry)


def task_aprea_nasdaq(smoke: bool = False, dry: bool = False):
    """NASDAQ 100 universe (n=54)."""
    return _run_named_literature("aprea-nasdaq", smoke, dry)
