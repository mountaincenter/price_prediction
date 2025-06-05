#!/usr/bin/env python3
"""
scripts/csv_to_center_shift_batch.py   v6.1  (2025-06-05)
────────────────────────────────────────────────────────
CHANGELOG — scripts/csv_to_center_shift_batch.py  （newest → oldest）
- 2025-06-05  v6.1 : 引数を廃止し自動バッチ化
    • data/prices/*.csv を総当たり
    • 各銘柄の diff.tex 生成は csv_to_center_shift_diff.process_one を呼び出し
    • 生成と同時に summary.tex を出力
- 2025-06-05  v6.0 : pandas 非依存・tabular 手組み化
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# diff モジュールを再利用
from csv_to_center_shift_diff import (
    process_one,        # diff.tex を 1 ファイル生成
    read_prices,
    calc_center_shift,
)

# --------------------------------------------------------------------------
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "data/prices"
OUT_DIR    = Path(__file__).resolve().parent.parent.parent / "data/analysis/center_shift"
SUMMARY_TEX = OUT_DIR / "summary.tex"

# ── モデル定数（batch 側でメトリクス計算に必要） ──────────────────────────
def compute_metrics(df: pd.DataFrame) -> tuple[float, float]:
    """MAE_5d(最終行) と HitRate_20d(最終行, %) を返す"""
    mae = df["MAE_5d"].iloc[-1]
    hit = df["HitRate_20d"].iloc[-1]      # diff.py 側で % 化済み
    return mae, hit

# ── LaTeX summary 生成 ──────────────────────────────────────────────────
def make_summary(rows: list[tuple[str, float, float, float, float]]) -> str:
    avg = lambda i: sum(r[i] for r in rows) / len(rows)
    rows.append(("Average", avg(1), avg(2), avg(3), avg(4)))

    def f(x: float) -> str: return f"{x:,.2f}"

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Code & Close & MAE\_5d & RelMAE[\%] & HitRate[\%] \\",
        r"\hline",
    ]
    for code, close, mae, rmae, hit in rows:
        lines.append(f"{code} & {f(close)} & {f(mae)} & {f(rmae)} & {f(hit)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines)

# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float, float, float]] = []
    for csv_path in sorted(PRICES_DIR.glob("*.csv")):
        # 1. diff.tex 生成
        process_one(csv_path, out_dir=OUT_DIR)

        # 2. 指標計算
        df = calc_center_shift(read_prices(csv_path))
        mae, hit = compute_metrics(df)
        close = df["Close"].iloc[-1]
        rows.append((csv_path.stem, close, mae, mae / close * 100, hit))

    # 3. summary.tex 出力
    SUMMARY_TEX.write_text(make_summary(rows), encoding="utf-8")
    print(f"✅ summary.tex 生成: {SUMMARY_TEX.relative_to(OUT_DIR.parent)}")

if __name__ == "__main__":
    main()
