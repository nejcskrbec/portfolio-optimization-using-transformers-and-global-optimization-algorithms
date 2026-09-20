#!/usr/bin/env python3
"""Price data loading: wide CSV files, yfinance download, the trimmed
missing-ticker resolver (explicit series overrides), and load_prices."""
from __future__ import annotations

import copy
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.utils.paths_utils import project_root


def _load_wide_file(path: Path, date_column: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "See the config/README. Exact-data literature configs intentionally "
            "expect the source-paper dataset to be placed locally."
        )

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    if date_column and date_column in df.columns:
        dcol = date_column
    else:
        candidates = [
            c for c in df.columns
            if str(c).strip().lower() in ("date", "dates", "time", "timestamp")
        ]
        dcol = candidates[0] if candidates else df.columns[0]

    idx = pd.to_datetime(df[dcol], errors="coerce")
    if idx.isna().mean() > 0.05:
        raise RuntimeError(
            f"Could not reliably parse date column '{dcol}' in {path}"
        )

    df = df.drop(columns=[dcol]).copy()
    df.index = idx
    df = df.loc[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=1, how="all")


def _require_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for data.source='yfinance'. "
            "Install it in the thesis environment or use data.source='csv'."
        ) from e


def _extract_close(raw, requested: list[str]) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=requested)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", level=-1, axis=1).copy()
        else:
            raise RuntimeError("Could not locate Close field in yfinance output")
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame(index=raw.index, columns=requested, dtype=float)
        close = raw[["Close"]].copy()
        if len(requested) == 1:
            close.columns = requested

    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested[0])

    return close.sort_index().reindex(columns=requested)


def _download_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    yf = _require_yfinance()
    raw = yf.download(
        tickers,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    return _extract_close(raw, tickers)


def _download_yfinance_symbol(symbol: str, start: str, end: str) -> pd.Series:
    """Download one symbol without raising merely because Yahoo returns empty."""
    try:
        df = _download_yfinance([symbol], start, end)
    except Exception:
        return pd.Series(dtype=float, name=symbol)
    if symbol not in df:
        return pd.Series(dtype=float, name=symbol)
    s = pd.to_numeric(df[symbol], errors="coerce")
    s.name = symbol
    return s.dropna()


def _load_series_override(spec: dict, start: str, end: str) -> pd.Series:
    """Load one explicitly supplied historical price series."""
    source = spec.get("url") or spec.get("path")
    if not source:
        raise ValueError("Series override requires 'url' or 'path'")

    if spec.get("url"):
        df = pd.read_csv(source)
    else:
        path = Path(source)
        if not path.is_absolute():
            path = project_root() / path
        if not path.exists():
            raise FileNotFoundError(f"Series override file not found: {path}")
        df = pd.read_csv(path)

    dcol = spec.get("date_column", "Date")
    vcol = spec.get("value_column", "Adj Close")
    if dcol not in df.columns or vcol not in df.columns:
        raise RuntimeError(
            f"Override {source} requires columns '{dcol}' and '{vcol}'. "
            f"Available={list(df.columns)}"
        )

    idx = pd.to_datetime(df[dcol], errors="coerce")
    vals = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=idx, dtype=float)
    s = s.loc[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.loc[
        (s.index >= pd.Timestamp(start)) &
        (s.index <= pd.Timestamp(end))
    ].dropna()


def _series_on_calendar(s: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(index=calendar, dtype=float)
    s = pd.to_numeric(s, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # No backward fill: a security that starts too late remains visibly invalid.
    return s.reindex(calendar).ffill()


def _coverage_problems(s: pd.Series, min_obs: int) -> list[str]:
    reasons = []
    if s is None or len(s) == 0:
        return ["missing"]
    if int(s.notna().sum()) < int(min_obs):
        reasons.append(f"obs<{min_obs}")
    if s.isna().any():
        reasons.append("nan_after_ffill")
    return reasons


def _resolve_missing_yfinance(
    requested: str,
    current: pd.Series,
    config: dict,
    calendar: pd.DatetimeIndex,
    start: str,
    end: str,
    min_obs: int,
):
    """Recover a ticker that yfinance could not serve with adequate coverage.

    The only repair channel is the config's own `series_overrides`, which names
    an explicit external CSV for a given symbol. Yahoo fails in two ways worth
    distinguishing, and only the first is caught automatically:

      * Nothing at all -- e.g. WBA, delisted in 2025 after going private, so no
        2006-2020 history is served. `_coverage_problems` flags it and the
        override supplies the series. Both Aprea configs do exactly this.
      * Valid-looking but silently truncated -- e.g. DOW starts at the Apr 2019
        spin-off, with earlier history under the old DOW and DWDP symbols.
        Nothing detects this: the data is internally consistent and passes the
        coverage check whenever the requested window begins after the break.
        Before adding a ticker with a merger/spin-off/rename in its past, verify
        the returned range covers the full window, and add a `series_overrides`
        entry if it does not.
    """
    dcfg = config["data"]
    report = {
        "requested_ticker": requested,
        "initial_problems": _coverage_problems(current, min_obs),
        "status": "unresolved",
    }

    # Explicit external series override is trusted and deterministic.
    override = dcfg.get("series_overrides", {}).get(requested)
    if override:
        try:
            raw = _load_series_override(override, start, end)
            s = _series_on_calendar(raw, calendar)
            if not _coverage_problems(s, min_obs):
                report.update({
                    "status": "resolved",
                    "method": "explicit_series_override",
                    "source": override.get("url") or override.get("path"),
                })
                return s, report
            report["override_problems"] = _coverage_problems(s, min_obs)
        except Exception as e:
            report["override_error"] = str(e)

    return current, report


def load_prices(config: dict) -> tuple[pd.DataFrame, list[str]]:
    dcfg = config["data"]
    source = str(dcfg.get("source", "yfinance")).lower()
    start = dcfg["start_date"]
    end = dcfg["end_date"]

    resolution_report = []

    if source == "csv":
        path = Path(dcfg["path"])
        if not path.is_absolute():
            path = project_root() / path
        close = _load_wide_file(path, dcfg.get("date_column"))
        configured = dcfg.get("tickers")
        if configured:
            tickers = [str(x) for x in configured]
            missing = [t for t in tickers if t not in close.columns]
            if missing:
                raise RuntimeError(f"CSV is missing configured tickers: {missing}")
            close = close[tickers]
        else:
            tickers = [str(x) for x in close.columns]

    elif source == "yfinance":
        tickers = [str(x) for x in dcfg["tickers"]]
        close = _download_yfinance(tickers, start, end)

        # Establish one common calendar from whatever Yahoo successfully returned.
        close = close.loc[
            (close.index >= pd.Timestamp(start)) &
            (close.index <= pd.Timestamp(end))
        ].sort_index()

        if close.empty:
            raise RuntimeError("yfinance returned no usable market calendar")

        # Forward-fill only internal gaps; no backward fill before listings.
        close = close.ffill()
        min_obs = int(dcfg.get("min_observations", 250))

        rcfg = dcfg.get("ticker_resolution", {})
        resolver_enabled = bool(rcfg.get("enabled", True))

        for ticker in tickers:
            current = (
                close[ticker]
                if ticker in close.columns
                else pd.Series(index=close.index, dtype=float)
            )
            problems = _coverage_problems(current, min_obs)

            if not problems:
                resolution_report.append({
                    "requested_ticker": ticker,
                    "status": "original",
                    "method": "yfinance_requested_symbol",
                    "observations": int(current.notna().sum()),
                })
                continue

            if not resolver_enabled:
                resolution_report.append({
                    "requested_ticker": ticker,
                    "status": "unresolved",
                    "method": "resolver_disabled",
                    "problems": problems,
                })
                continue

            print(
                f"[ticker-resolver] {ticker}: {', '.join(problems)} -> "
                "trying explicit series override"
            )
            resolved, report = _resolve_missing_yfinance(
                ticker,
                current=current,
                config=config,
                calendar=close.index,
                start=start,
                end=end,
                min_obs=min_obs,
            )
            close[ticker] = resolved
            report["observations"] = int(resolved.notna().sum())
            report["final_problems"] = _coverage_problems(resolved, min_obs)
            resolution_report.append(report)

            if report.get("status") == "resolved":
                print(
                    f"[ticker-resolver] {ticker}: resolved via "
                    f"{report.get('method')}"
                    + (
                        f" -> {report.get('resolved_symbol')}"
                        if report.get("resolved_symbol") else ""
                    )
                )
            else:
                print(f"[ticker-resolver] {ticker}: unresolved")

    else:
        raise ValueError(f"Unsupported data source: {source}")

    close = close.loc[
        (close.index >= pd.Timestamp(start)) &
        (close.index <= pd.Timestamp(end))
    ].copy()
    close = close.ffill()

    min_obs = int(dcfg.get("min_observations", 250))
    strict = bool(dcfg.get("strict_universe", True))
    valid = []
    rejected = {}

    for t in tickers:
        if t not in close:
            rejected[t] = ["missing"]
            continue
        reasons = _coverage_problems(close[t], min_obs)
        if reasons:
            rejected[t] = reasons
        else:
            valid.append(t)

    config.setdefault("_runtime", {})["ticker_resolution_report"] = resolution_report

    if rejected:
        msg = "\n".join(f"  - {t}: {', '.join(r)}" for t, r in rejected.items())
        if strict:
            raise RuntimeError(
                "Incomplete investment universe after ticker resolution:\n" + msg
            )
        print("[data] dropped assets:\n" + msg)

    close = close[valid]
    expected_n = dcfg.get("expected_assets")
    if expected_n is not None and len(valid) != int(expected_n):
        msg = (
            f"Expected {expected_n} assets for source-paper universe, got {len(valid)}."
        )
        if bool(dcfg.get("enforce_expected_assets", False)):
            raise RuntimeError(msg)
        print("[warning]", msg)

    if len(valid) < int(config["portfolio"]["cardinality_K"]):
        raise RuntimeError(
            f"Only {len(valid)} valid assets but K={config['portfolio']['cardinality_K']}"
        )
    return close, valid

