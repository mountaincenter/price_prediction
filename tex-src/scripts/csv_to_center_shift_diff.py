#!/usr/bin/env python3
"""
scripts/csv_to_center_shift_diff.py   v2.14  (2025-06-06)
────────────────────────────────────────────────────────
- CHANGELOG — scripts/csv_to_center_shift_diff.py  （newest → oldest）
- 2025-06-06  v2.14: diff テーブル末尾に改行を付与
- 2025-06-06  v2.13: "code:" 表示を表上部のキャプションへ移動
- 2025-06-06  v2.12: α_t/λ_shift/Δα_t 列を追加し code 表示を挿入
- 2025-06-06  v2.11: Average 行の下に Median 行を追加
- 2025-06-05  v2.10: HitRate 改善アルゴリズム導入
- 2025-06-05  v2.9 : Phase 2 専用スクリプトである旨を明記
- 2025-06-05  v2.8 : process_one が Path を返すよう変更
- 2025-06-05  v2.7 : 初期5日間の $S_t$ 無効化
- 2025-06-05  v2.6 : FutureWarning 非表示
- 2025-06-05  v2.5 : ルート基準でパスを解決
- 2025-06-05  v2.4 : LaTeX 文字列中の生 '\\n' を排除／HitRate[%] を 0.60→60.00 表示
- 2025-06-05  v2.3 : MM-DD 表示 + threeparttable 脚注を同枠内に配置
- 2025-06-05  v2.2 : f-string 内 '}' → '}}' ダブルエスケープ
- 2025-06-05  v2.1 : v2.0 の括弧／{} 構文エラー修正
- 2025-06-05  v2.0 : 列幅縮小＋RelMAE[%] 追加ほか
- 2025-06-05  v1.5 : Average 行追加 & 一括処理デフォルト
- 2025-06-04  v1.4 : 評価列追加
- 2025-06-04  v1.3 : LaTeX ヘッダ数式化
- 2025-06-04  v1.2 : Date フォーマット例外処理
- 2025-06-04  v1.1 : 列簡略化
- 2025-06-04  v1.0 : 初版
"""

# Phase 2 の中心シフト計算を行い、diff テーブルを生成する。
# 詳細な数式は tex-src/center_shift/phase2.tex を参照。

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
KAPPA_BUCKETS = [
    (0.00, 0.01, 0.05),
    (0.01, 0.02, 0.10),
    (0.02, 0.04, 0.15),
    (0.04, np.inf, 0.20),
]
L_INIT, L_MIN, L_MAX = 0.94, 0.90, 0.98
ETA = 0.01
VAR_EPS = 1e-8

NUM_ROWS = 30                      # 最新 30 行 + Average
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/center_shift"
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/prices"

# ──────────────────────────────────────────────────────────────
def resolve_csv(raw: Path) -> Path:
    if raw.exists():
        return raw.resolve()
    alt = PRICES_DIR / raw.name
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(raw)

def read_prices(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv).rename(columns=lambda c: c.strip().replace("　", ""))
    rename = {
        "Date": "Date", "日付": "Date",
        "High": "High", "高値": "High",
        "Low":  "Low",  "安値": "Low",
        "Close":"Close","終値": "Close",
    }
    df = df.rename(columns=rename, errors="ignore")
    if {"Date", "High", "Low", "Close"} - set(df.columns):
        raise KeyError(f"{csv.name}: Date / High / Low / Close が不足")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["DispDate"] = df["Date"].dt.strftime("%m-%d")
    for c in ["High", "Low", "Close"]:
        df[c] = df[c].replace({",": ""}, regex=True).astype(float)
    return df

# ──────────────────────────────────────────────────────────────
def kappa_sigma(s: float) -> float:
    for lo, hi, v in KAPPA_BUCKETS:
        if lo <= s < hi:
            return v
    return 0.20

def calc_center_shift(df: pd.DataFrame, phase: int = 2) -> pd.DataFrame:
    n = len(df)
    cl = df["Close"].values
    dcl = np.zeros(n); dcl[1:] = np.log(cl[1:] / cl[:-1])

    sig = np.zeros(n); lam = np.full(n, L_INIT)
    kap = np.zeros(n); alp = np.zeros(n)
    dalp = np.zeros(n)
    S = np.zeros(n); ma3 = np.zeros(n)
    var = max(VAR_EPS, (np.pi / 2) * abs(dcl[0])) ** 2

    for t in range(n):
        if t:
            var = max(lam[t-1]*var + (1-lam[t-1])*dcl[t]**2, VAR_EPS)
        sig[t] = sqrt(var)
        kap[t] = 0.20 if phase == 0 else kappa_sigma(sig[t])
        if t:
            ma3[t] = dcl[max(0, t-3):t].mean()
        if phase == 2:
            S[t] = np.sign(ma3[t])
            if abs(ma3[t]) < 0.5 * sig[t]:
                S[t] = 0
        else:
            S[t] = np.sign(dcl[t-1]) if t else 0
        if t < 5:
            S[t] = 0
        alp[t] = kap[t] * S[t]
        if t:
            dalp[t] = alp[t] - alp[t-1]

        if phase == 2 and t >= 31:
            e = dcl[t-30:t]**2 - sig[t-30:t]**2
            g = -(2/30) * np.sum(e * sig[t-30:t]**2)
            lam[t] = np.clip(lam[t-1] - ETA*np.clip(g, -10, 10), L_MIN, L_MAX)
        else:
            lam[t] = lam[t-1] if t else L_INIT

    out = pd.DataFrame({
        "Date": df["DispDate"],
        r"$\kappa(\sigma)$": kap,
        "High": df["High"],
        "Low":  df["Low"],
        r"$\alpha_t$": alp,
        r"$\lambda_{\text{shift}}$": lam,
        r"$\Delta\alpha_t$": dalp,
        r"$\sigma_t^{\mathrm{shift}}$": sig,
        "Close": df["Close"]
    })
    out["B_{t-1}"] = (out["High"].shift(1) + out["Low"].shift(1)) / 2
    out["C_pred"]  = out["B_{t-1}"] * (1 + out[r"$\alpha_t$"])
    out["C_real"]  = (out["High"] + out["Low"]) / 2
    out["C_diff"]  = out["C_pred"] - out["C_real"]

    out["C_diff_sign"] = np.sign(out["C_diff"])
    out["Norm_err"]    = np.abs(out["C_diff"]) / out[r"$\sigma_t^{\mathrm{shift}}$"]
    out["MAE_5d"]      = out["C_diff"].abs().rolling(5, min_periods=1).mean()
    out["RelMAE"]      = out["MAE_5d"] / out["Close"] * 100       # %
    hit = (np.sign(out[r"$\alpha_t$"]) ==
           np.sign(out["C_real"] - out["B_{t-1}"])).astype(int)
    out["HitRate_20d"] = hit.rolling(20, min_periods=1).mean() * 100  # %
    return out

# ──────────────────────────────────────────────────────────────
def make_table(df: pd.DataFrame, code: str = "") -> str:
    dfn = df.tail(NUM_ROWS).iloc[::-1].reset_index(drop=True)

    avg = {"Date": "Average"}
    med = {"Date": "Median"}
    for c in [r"$\kappa(\sigma)$","B_{t-1}","C_pred","C_real","C_diff",
              "C_diff_sign","Norm_err","MAE_5d","RelMAE","HitRate_20d"]:
        vals = dfn[c].astype(float)
        avg[c] = vals.mean()
        med[c] = np.median(vals)
    dfn = pd.concat([dfn, pd.DataFrame([avg, med])], ignore_index=True)

    cols_src = [
        "Date",
        r"$\kappa(\sigma)$",
        "B_{t-1}",
        "C_pred",
        "C_real",
        "C_diff",
        "C_diff_sign",
        "Norm_err",
        r"$\alpha_t$",
        r"$\lambda_{\text{shift}}$",
        r"$\Delta\alpha_t$",
        "MAE_5d",
        "RelMAE",
        "HitRate_20d",
    ]
    header = {
        r"$\kappa(\sigma)$": r"$\kappa$",
        "B_{t-1}":            r"$B$",
        "C_pred":             r"$C_p$",
        "C_real":             r"$C_r$",
        "C_diff":             r"$C_\Delta$",
        "C_diff_sign":        r"$\mathrm{sgn}\,C_\Delta$",
        "Norm_err":           r"$|C_\Delta|/\sigma$",
        r"$\alpha_t$":        r"$\alpha_t$",
        r"$\lambda_{\text{shift}}$": r"$\lambda$",
        r"$\Delta\alpha_t$": r"$\Delta\alpha$",
        "MAE_5d":             r"$\mathrm{MAE}_5$",
        "RelMAE":             r"$\mathrm{RMAE}$",
        "HitRate_20d":        r"$\mathrm{HR}_{20}[\%]$",
    }
    cols = [header.get(c, c) for c in cols_src]

    def fmt(v, col):
        if col == "Date":
            return v
        if pd.isna(v):
            return "--"
        if col in {r"$\kappa$", r"$\alpha_t$", r"$\lambda$", r"$\Delta\alpha$"}:
            return f"{v:.2f}"
        if col in {r"$\mathrm{RMAE}$", r"$\mathrm{HR}_{20}[\%]$"}:
            return f"{v:.2f}"
        return f"{v:.1f}"

    disp = pd.DataFrame({
        cols[i]: [fmt(v, cols[i]) for v in dfn[cols_src[i]]]
        for i in range(len(cols))
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        fmt_str = "l" + "r" * (len(cols) - 1)
        latex_body = disp.to_latex(
            index=False, escape=False, column_format=fmt_str
        )

    footnote_lines = [
        r"\begin{tablenotes}\footnotesize",
        r"\item $\kappa=\kappa(\sigma)$, $B=B_{t-1}$, "
        r"$C_p=C_{\text{pred}}$, $C_r=C_{\text{real}}$, "
        r"$C_\Delta=C_{\text{diff}}$, "
        r"$\mathrm{sgn}\,C_\Delta=\operatorname{sign}(C_{\text{diff}})$, "
        r"$|C_\Delta|/\sigma=\dfrac{|C_{\text{diff}}|}{\sigma_t^{\text{shift}}}$, "
        r"$\mathrm{MAE}_5=\mathrm{MAE}_{5\text{d}}$, "
        r"$\mathrm{RMAE}= \mathrm{MAE}_5 / \text{Close}$, "
        r"$\mathrm{HR}_{20}=\mathrm{HitRate}_{20\text{d}}$, ",
        r"$\lambda_{\text{shift}}=\lambda_t$, ",
        r"$\Delta\alpha_t=\alpha_t-\alpha_{t-1}$.",
        r"\end{tablenotes}"
    ]
    footnote = "\n".join(footnote_lines)

    parts = [
        (f"\\noindent\\textbf{{code:{code}}}\\" if code else ""),
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}%",
        r"\begin{threeparttable}",
    ]
    if code:
        parts.append(rf"\caption*{{\textbf{{code:{code}}}}}")
    parts += [
        r"\resizebox{\textwidth}{!}{%",
        latex_body.rstrip(),
        r"}%",
        footnote,
        r"\end{threeparttable}",
        r"\endgroup"
    ]
    return "\n".join(parts) + "\n"

# ──────────────────────────────────────────────────────────────
def process_one(csv: Path, out_dir: Path = OUT_DIR) -> Path:
    """csv を処理して diff.tex を生成し、そのパスを返す"""
    code = csv.stem
    tex = make_table(calc_center_shift(read_prices(csv)), code)
    out = out_dir / f"{code}_diff.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out

# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="center_shift diff table (30 d ＋ Average)"
    )
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="個別 CSV（省略時は data/prices/*.csv 一括処理）"
    )
    args = parser.parse_args()

    if args.csv is None:
        for p in sorted(PRICES_DIR.glob("*.csv")):
            out = process_one(p)
            print(f"✅ {p.stem} → {out.relative_to(OUT_DIR.parent.parent)}")
    else:
        out = process_one(resolve_csv(args.csv))
        print(f"✅ {args.csv.stem} → {out.relative_to(OUT_DIR.parent.parent)}")

if __name__ == "__main__":
    main()
