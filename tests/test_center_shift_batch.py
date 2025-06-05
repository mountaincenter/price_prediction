#!/usr/bin/env python3
""" tests/test_center_shift_batch.py
  v1.0  (2025-06-05)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-05  新規作成
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tex-src" / "scripts"))

from csv_to_center_shift_diff import read_prices, calc_center_shift
from csv_to_center_shift_batch import compute_metrics, make_summary

PRICES_DIR = Path(__file__).resolve().parent / "fixtures" / "prices"

def test_compute_metrics_values():
    df_raw = read_prices(PRICES_DIR / "1321.csv")
    df = calc_center_shift(df_raw, phase=2)
    mae, rmae, hit = compute_metrics(df)
    assert mae == df["MAE_5d"].iloc[-1]
    assert rmae == df["RelMAE"].iloc[-1]
    assert hit == df["HitRate_20d"].iloc[-1]

def test_make_summary_output():
    df_raw = read_prices(PRICES_DIR / "1321.csv")
    df0 = calc_center_shift(df_raw, phase=0)
    df1 = calc_center_shift(df_raw, phase=1)
    df2 = calc_center_shift(df_raw, phase=2)
    mae, r2, h2 = compute_metrics(df2)
    _, r0, h0 = compute_metrics(df0)
    _, r1, h1 = compute_metrics(df1)
    close = df2["Close"].iloc[-1]
    rows = [("1321", close, mae, r0, r1, r2, h0, h1, h2)]
    summary = make_summary(rows)
    assert "Average" in summary
    assert "tabular" in summary
