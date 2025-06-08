#!/usr/bin/env python3
"""scripts/csv_to_event_batch.py   v1.0  (2025-06-10)
────────────────────────────────────────────────────────
- CHANGELOG — scripts/csv_to_event_batch.py  （newest → oldest）
- 2025-06-10  v1.0 : 初版
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from csv_to_event_diff import (
    process_one,
    read_prices,
    calc_event_beta,
    ETA,
    L_INIT,
    L_MIN,
    L_MAX,
)

PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/prices"
OUT_DIR    = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/event"

SUMMARY_TEX = OUT_DIR / "summary.tex"

# ── モデル定数 ────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """MAE_5d, RelMAE, HitRate_20d (各最終行, %) を返す"""
    mae = df["MAE_5d"].iloc[-1]
    rmae = df["RelMAE"].iloc[-1]
    hit = df["HitRate_20d"].iloc[-1]
    return mae, rmae, hit

# ── LaTeX summary 生成 ───────────────────────────────────────────────
def make_summary(rows: list[tuple[str, float, float, float, float]]) -> str:
    avg = lambda i: sum(r[i] for r in rows) / len(rows)
    med = lambda i: float(np.median([r[i] for r in rows]))
    rows.append(("Average", avg(1), avg(2), avg(3), avg(4)))
    rows.append(("Median", med(1), med(2), med(3), med(4)))

    def f(x: float) -> str:
        return f"{x:,.2f}"

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Code & Close & MAE\_5d & RelMAE[\%] & HitRate[\%] \\",
        r"\hline",
    ]
    for code, close, mae, rmae, hit in rows:
        lines.append(f"{code} & {f(close)} & {f(mae)} & {f(rmae)} & {f(hit)} \\")
    lines += [r"\hline", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines)

# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="event batch processor")
    p.add_argument("--eta", type=float, default=ETA, help="学習率 η")
    p.add_argument("--init-lambda", type=float, default=L_INIT, help="初期 λ_shift")
    p.add_argument("--min-lambda", type=float, default=L_MIN, help="最小 λ_shift")
    p.add_argument("--max-lambda", type=float, default=L_MAX, help="最大 λ_shift")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float, float, float]] = []
    for csv_path in sorted(PRICES_DIR.glob("*.csv")):
        out = process_one(
            csv_path,
            out_dir=OUT_DIR,
            eta=args.eta,
            l_init=args.init_lambda,
            l_min=args.min_lambda,
            l_max=args.max_lambda,
        )
        print(f"✅ {csv_path.stem} → {out.relative_to(OUT_DIR.parent.parent)}")

        raw = read_prices(csv_path)
        df = calc_event_beta(
            raw,
            eta=args.eta,
            l_init=args.init_lambda,
            l_min=args.min_lambda,
            l_max=args.max_lambda,
        )
        mae, rmae, hit = compute_metrics(df)
        close = df["Close"].iloc[-1]
        rows.append((csv_path.stem, close, mae, rmae, hit))

    SUMMARY_TEX.write_text(make_summary(rows), encoding="utf-8")
    print(f"✅ summary.tex 生成: {SUMMARY_TEX.relative_to(OUT_DIR.parent)}")

if __name__ == "__main__":
    main()
