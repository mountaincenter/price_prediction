#!/usr/bin/env python3
""" tests/test_center_shift_diff.py
  v1.0  (2025-06-05)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-05  新規作成
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tex-src" / "scripts"))

from csv_to_center_shift_diff import read_prices, calc_center_shift, make_table, process_one

PRICES_DIR = Path(__file__).resolve().parent / "fixtures" / "prices"

def test_read_prices_sort_and_columns():
    csv_path = PRICES_DIR / "1321.csv"
    df = read_prices(csv_path)
    assert df["Date"].is_monotonic_increasing
    for col in ["High", "Low", "Close"]:
        assert col in df.columns

def test_calc_center_shift_shape_and_columns():
    csv_path = PRICES_DIR / "1321.csv"
    df_raw = read_prices(csv_path)
    df = calc_center_shift(df_raw)
    assert len(df) == len(df_raw)
    for col in ["MAE_5d", "RelMAE", "HitRate_20d"]:
        assert col in df.columns

def test_make_table_contains_latex():
    csv_path = PRICES_DIR / "1321.csv"
    df_raw = read_prices(csv_path)
    df = calc_center_shift(df_raw)
    tex = make_table(df)
    assert "\\begin{threeparttable}" in tex

def test_process_one_creates_file(tmp_path):
    csv_path = PRICES_DIR / "1321.csv"
    out = process_one(csv_path, out_dir=tmp_path)
    assert out.exists()
    assert "tabular" in out.read_text()
