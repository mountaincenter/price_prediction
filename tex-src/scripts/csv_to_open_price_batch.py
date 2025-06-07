#!/usr/bin/env python3
"""
scripts/csv_to_open_price_batch.py   v1.1  (2025-06-07)
────────────────────────────────────────────────────────
- CHANGELOG — scripts/csv_to_open_price_batch.py  （newest → oldest）
- 2025-06-07  v1.1 : Open 用サマリー表に対応
- 2025-06-07  v1.0 : 初版（center_shift_batch.py から派生）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# diff モジュールを再利用
from csv_to_open_price_diff import (
    process_one,        # diff.tex を 1 ファイル生成
    read_prices,
    calc_open_price,
    ETA,
    L_INIT,
    L_MIN,
    L_MAX,
)

# --------------------------------------------------------------------------
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/prices"
OUT_DIR    = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/open_price"

SUMMARY_TEX = OUT_DIR / "summary.tex"

# ── モデル定数（batch 側でメトリクス計算に必要） ──────────────────────────
def compute_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """MAE_5d, RelMAE, HitRate_20d (各最終行, %) を返す"""
    mae = df["MAE_5d"].iloc[-1]
    rmae = df["RelMAE"].iloc[-1]
    hit = df["HitRate_20d"].iloc[-1]
    return mae, rmae, hit

# ── LaTeX summary 生成 ──────────────────────────────────────────────────
def make_summary(rows: list[tuple[str, float, float, float, float, float, float, float, float]]) -> str:
    avg = lambda i: sum(r[i] for r in rows) / len(rows)
    med = lambda i: float(np.median([r[i] for r in rows]))
    rows.append(("Average", avg(1), avg(2), avg(3), avg(4), avg(5), avg(6), avg(7), avg(8)))
    rows.append(("Median", med(1), med(2), med(3), med(4), med(5), med(6), med(7), med(8)))

    def f(x: float) -> str: return f"{x:,.2f}"

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\hline",
        r"Code & Open & MAE\_5d & RelMAE$^{ph0}$[\%] & RelMAE$^{ph1}$[\%] & RelMAE$^{ph2}$[\%] & HitRate$^{ph0}$[\%] & HitRate$^{ph1}$[\%] & HitRate$^{ph2}$[\%] \\",
        r"\hline",
    ]
    for code, open_, mae, r0, r1, r2, h0, h1, h2 in rows:
        lines.append(f"{code} & {f(open_)} & {f(mae)} & {f(r0)} & {f(r1)} & {f(r2)} & {f(h0)} & {f(h1)} & {f(h2)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines)

# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="open_price batch processor")
    p.add_argument("--eta", type=float, default=ETA, help="学習率 η")
    p.add_argument("--init-lambda", type=float, default=L_INIT, help="初期 λ_shift")
    p.add_argument("--min-lambda", type=float, default=L_MIN, help="最小 λ_shift")
    p.add_argument("--max-lambda", type=float, default=L_MAX, help="最大 λ_shift")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float, float, float, float, float, float, float]] = []
    for csv_path in sorted(PRICES_DIR.glob("*.csv")):
        # 1. diff.tex 生成
        out = process_one(
            csv_path,
            out_dir=OUT_DIR,
            eta=args.eta,
            l_init=args.init_lambda,
            l_min=args.min_lambda,
            l_max=args.max_lambda,
        )
        print(f"✅ {csv_path.stem} → {out.relative_to(OUT_DIR.parent.parent)}")

        # 2. 指標計算
        raw = read_prices(csv_path)
        df0 = calc_open_price(raw, phase=0, eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda)
        df1 = calc_open_price(raw, phase=1, eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda)
        df2 = calc_open_price(raw, phase=2, eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda)

        mae2, r2, h2 = compute_metrics(df2)
        _, r0, h0 = compute_metrics(df0)
        _, r1, h1 = compute_metrics(df1)

        open_ = df2["Open"].iloc[-1]
        rows.append((csv_path.stem, open_, mae2, r0, r1, r2, h0, h1, h2))

    # 3. summary.tex 出力
    SUMMARY_TEX.write_text(make_summary(rows), encoding="utf-8")
    print(f"✅ summary.tex 生成: {SUMMARY_TEX.relative_to(OUT_DIR.parent)}")

if __name__ == "__main__":
    main()
