#!/usr/bin/env python3
"""
scripts/csv_to_center_shift_batch.py   v6.13  (2025-06-05)
────────────────────────────────────────────────────────
- CHANGELOG — scripts/csv_to_center_shift_batch.py  （newest → oldest）
- 2025-06-13  v6.13: MAE_10d 指標を計算
- 2025-06-13  v6.12: マクロイベント日をイベント値2として処理
- 2025-06-06  v6.11: diff テーブルを λ 固定で3ページ出力
- 2025-06-06  v6.10: η / λ を diff.py と同様に指定可能に
- 2025-06-06  v6.9 : 新スケーリング式に対応
- 2025-06-06  v6.8 : summary.tex に Median 行を追加
- 2025-06-05  v6.7 : HitRate 改善アルゴリズムに対応
- 2025-06-05  v6.6 : LaTeX ヘッダの上付記号を数式モードへ
- 2025-06-05  v6.5 : RelMAE/HitRate 各 Phase 列を追加
- 2025-06-05  v6.4 : diff 生成メッセージを main で出力
- 2025-06-05  v6.3 : 初期5日間 $S_t$ 無効化に対応
- 2025-06-05  v6.2 : ルート基準でパスを解決
- 2025-06-05  v6.1 : 引数を廃止し自動バッチ化
    • data/prices/*.csv を総当たり
    • 各銘柄の diff.tex 生成は csv_to_center_shift_diff.process_one を呼び出し
    • 生成と同時に summary.tex を出力
- 2025-06-05  v6.0 : pandas 非依存・tabular 手組み化
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# diff モジュールを再利用
from csv_to_center_shift_diff import (
    process_one,        # diff.tex を 1 ファイル生成
    read_prices,
    calc_center_shift,
    ETA,
    L_INIT,
    L_MIN,
    L_MAX,
    EVENTS_CSV,
)

# --------------------------------------------------------------------------
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/prices"
OUT_DIR    = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/center_shift"

SUMMARY_TEX = OUT_DIR / "summary.tex"

# ── モデル定数（batch 側でメトリクス計算に必要） ──────────────────────────
def compute_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """MAE_10d, RelMAE, HitRate_20d (各最終行, %) を返す"""
    mae = df["C_diff"].abs().rolling(10, min_periods=1).mean().dropna().iloc[-1]
    rmae = (
        df["C_diff"].abs().rolling(5, min_periods=1).mean()
        .div(df["Close"])
        .mul(100)
        .dropna()
        .iloc[-1]
    )
    hit = (
        (np.sign(df[r"$\alpha_t$"]) == np.sign(df["C_real"] - df["B_{t-1}"]))
        .astype(int)
        .rolling(20, min_periods=1)
        .mean()
        .dropna()
        .iloc[-1]
        * 100
    )
    return float(mae), float(rmae), float(hit)

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
        r"Code & Close & MAE\_10d & RelMAE$^{ph0}$[\%] & RelMAE$^{ph1}$[\%] & RelMAE$^{ph2}$[\%] & HitRate$^{ph0}$[\%] & HitRate$^{ph1}$[\%] & HitRate$^{ph2}$[\%] \\",
        r"\hline",
    ]
    for code, close, mae, r0, r1, r2, h0, h1, h2 in rows:
        lines.append(f"{code} & {f(close)} & {f(mae)} & {f(r0)} & {f(r1)} & {f(r2)} & {f(h0)} & {f(h1)} & {f(h2)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines)

# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="center_shift batch processor")
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
            events_csv=EVENTS_CSV,
        )
        print(f"✅ {csv_path.stem} → {out.relative_to(OUT_DIR.parent.parent)}")

        # 2. 指標計算
        raw = read_prices(csv_path)
        df0 = calc_center_shift(raw, phase=0, eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda, events_csv=EVENTS_CSV)
        df1 = calc_center_shift(raw, phase=1, eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda, events_csv=EVENTS_CSV)
        df2 = calc_center_shift(raw, phase=2, eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda, events_csv=EVENTS_CSV)

        mae2, r2, h2 = compute_metrics(df2)
        _, r0, h0 = compute_metrics(df0)
        _, r1, h1 = compute_metrics(df1)

        close = df2["Close"].iloc[-1]
        rows.append((csv_path.stem, close, mae2, r0, r1, r2, h0, h1, h2))

    # 3. summary.tex 出力
    SUMMARY_TEX.write_text(make_summary(rows), encoding="utf-8")
    print(f"✅ summary.tex 生成: {SUMMARY_TEX.relative_to(OUT_DIR.parent)}")

if __name__ == "__main__":
    main()
