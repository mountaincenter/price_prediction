#!/usr/bin/env python3
"""
scripts/csv_to_center_shift.py   v1.7  (2025-06-04)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-05  v1.7 : ルート基準でパスを解決
- 2025-06-04  v1.6 : 抽出期間を 14 → **63 営業日**（約 3 ヶ月）へ拡張
                    • 表示桁数 (%.5f) は維持
- 2025-06-04  v1.5 : 4 分割 → 1 表統合、\scriptsize + \resizebox
- 2025-06-04  v1.4 : wrap() を実改行化（LaTeX エラー修正）
- 2025-06-04  v1.3 : Date / Close 列バリアント対応
- 2025-06-04  v1.2 : \scriptsize フォント
- 2025-06-03  v1.0 : 初版
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
KAPPA_BUCKETS = [
    (0.00, 0.01, 0.05),
    (0.01, 0.02, 0.10),
    (0.02, 0.04, 0.15),
    (0.04, np.inf, 0.20),
]
LAMBDA_INIT = 0.94
LAMBDA_MIN, LAMBDA_MAX = 0.90, 0.98
ETA = 0.01
VAR_EPS = 1e-8
NUM_ROWS = 63  # ← ここだけで期間を調整できます

# ──────────────────────────────────────────────────────────────
def resolve_csv(raw: Path) -> Path:
    if raw.exists():
        return raw.resolve()
    alt = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data" / "prices" / raw.name
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(raw)

def read_prices(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv).rename(columns=lambda c: c.strip().replace("　", ""))

    rename = {
        "Date": "Date", "date": "Date", "日付": "Date",
        "Close": "Close", "close": "Close", "終値": "Close",
    }
    df = df.rename(columns=rename, errors="ignore")
    if {"Date", "Close"} - set(df.columns):
        raise KeyError("Date / Close 列が見つかりません")

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = (
        df["Close"].replace({",": ""}, regex=True).pipe(pd.to_numeric, errors="coerce")
    )
    return df.dropna(subset=["Close"])

# ──────────────────────────────────────────────────────────────
def kappa_sigma(sig: float) -> float:
    for lo, hi, val in KAPPA_BUCKETS:
        if lo <= sig < hi:
            return val
    return 0.20

def center_shift(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    cl = df["Close"].values
    dcl = np.zeros(n)
    dcl[1:] = np.log(cl[1:] / cl[:-1])

    sig = np.zeros(n)
    lam = np.full(n, LAMBDA_INIT)
    kap = np.zeros(n)
    alp = np.zeros(n)
    S = np.zeros(n)

    var_prev = max(VAR_EPS, (np.pi / 2) * abs(dcl[0])) ** 2

    for t in range(n):
        if t > 0:
            var_prev = (
                lam[t - 1] * var_prev + (1 - lam[t - 1]) * dcl[t] ** 2
            )
            var_prev = max(var_prev, VAR_EPS)
        sig[t] = sqrt(var_prev)

        S[t] = np.sign(dcl[t - 1]) if t > 0 else 0
        kap[t] = kappa_sigma(sig[t])
        alp[t] = kap[t] * S[t]

        if t >= 31:
            e = dcl[t - 30 : t] ** 2 - sig[t - 30 : t] ** 2
            g = -(2 / 30) * np.sum(e * sig[t - 30 : t] ** 2)
            lam[t] = np.clip(
                lam[t - 1] - ETA * np.clip(g, -10, 10),
                LAMBDA_MIN,
                LAMBDA_MAX,
            )
        else:
            lam[t] = lam[t - 1] if t > 0 else LAMBDA_INIT

    return pd.DataFrame(
        {
            "Date": df["Date"],
            r"$S_t$": S,
            r"$\alpha_t$": alp,
            r"$\sigma_t^{\mathrm{shift}}$": sig,
            r"$\kappa(\sigma)$": kap,
            r"$\lambda_{\mathrm{shift}}$": lam,
        }
    )

# ──────────────────────────────────────────────────────────────
def make_table(df: pd.DataFrame) -> str:
    # 最新 NUM_ROWS 行を新しい順で表示
    dfn = df.tail(NUM_ROWS).iloc[::-1].reset_index(drop=True)
    fmt = {"index": False, "escape": False, "float_format": "%.10f"}

    tab = dfn[
        [
            "Date",
            r"$S_t$",
            r"$\alpha_t$",
            r"$\sigma_t^{\mathrm{shift}}$",
            r"$\kappa(\sigma)$",
            r"$\lambda_{\mathrm{shift}}$",
        ]
    ].to_latex(**fmt)

    return (
        "\\begingroup\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tab.strip()}\n"
        "}%\n"
        "\\endgroup\n"
    )

# ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="center_shift Phase0–2 統合テーブル生成 (63d)"
    )
    ap.add_argument("csv", type=Path, help="data/prices/<code>.csv")
    ap.add_argument("--out", "-o", type=Path, help="出力先を明示指定")
    args = ap.parse_args()

    csv = resolve_csv(args.csv)
    code = csv.stem

    tex = make_table(center_shift(read_prices(csv)))

    out = (
        args.out.resolve()
        if args.out
        else (
            Path(__file__).resolve().parent.parent.parent
            / "tex-src" / "data" / "analysis" / "center_shift" / f"{code}.tex"

        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"✅ 生成完了: {out}")

if __name__ == "__main__":
    main()
