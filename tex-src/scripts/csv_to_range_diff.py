#!/usr/bin/env python3
"""
scripts/csv_to_range_diff.py   v1.0  (2025-06-11)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-11  v1.0 : 初版
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
L_INIT, L_MIN, L_MAX = 0.90, 0.80, 0.99
ETA_SIG = 0.5
ETA_VOL = 0.4
ETA = 0.01
VAR_EPS = 1e-8

NUM_ROWS = 30
VARIANT_LAMBDAS = [(0.80, "minimum"), (0.90, "default"), (0.99, "maximum")]
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/range"
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
        "Low": "Low", "安値": "Low",
        "Close": "Close", "終値": "Close",
        "Volume": "Volume", "出来高": "Volume",
    }
    df = df.rename(columns=rename, errors="ignore")
    if {"Date", "High", "Low", "Close"} - set(df.columns):
        raise KeyError(f"{csv.name}: Date/High/Low/Close が不足")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["DispDate"] = df["Date"].dt.strftime("%m-%d")
    for c in ["High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = df[c].replace({",": ""}, regex=True).astype(float)
    return df

# ──────────────────────────────────────────────────────────────

def calc_range(
    df: pd.DataFrame,
    *,
    eta: float = ETA,
    l_init: float = L_INIT,
    l_min: float = L_MIN,
    l_max: float = L_MAX,
) -> pd.DataFrame:
    n = len(df)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    vol = df.get("Volume", pd.Series(np.ones(n))).shift(1).fillna(0).values

    dcl = np.zeros(n); dcl[1:] = np.log(close[1:] / close[:-1])
    sig = np.zeros(n); lam = np.full(n, l_init)
    var = max(VAR_EPS, (np.pi / 2) * abs(dcl[0])) ** 2
    for t in range(n):
        if t:
            var = max(lam[t-1]*var + (1-lam[t-1])*dcl[t]**2, VAR_EPS)
        sig[t] = sqrt(var)
        if t >= 31:
            e = dcl[t-30:t]**2 - sig[t-30:t]**2
            g = -(2/30)*np.sum(e * sig[t-30:t]**2)
            lam[t] = np.clip(lam[t-1] - eta*np.clip(g, -10, 10), l_min, l_max)
        else:
            lam[t] = lam[t-1] if t else l_init

    m_real = (high - low) / 2

    beta1 = np.ones(n)
    for t in range(n):
        if t < 1:
            beta1[t] = 1.0
        else:
            hist = m_real[max(0, t-63):t]
            iqr = np.percentile(hist, 75) - np.percentile(hist, 25)
            iqr = max(iqr, 1e-4)
            beta1[t] = np.clip(1/iqr, 0.20, 5.0)

    beta2 = np.ones(n)
    for t in range(n):
        mean_sig = sig[max(0, t-63):t].mean() if t else sig[0]
        r_sig = sig[t] / mean_sig if mean_sig > 0 else 1.0
        r_sig = np.clip(r_sig, 0.3, 3.0)
        beta2[t] = beta1[t] * (r_sig ** ETA_SIG)

    beta3 = np.ones(n)
    vol_avg = pd.Series(df.get("Volume", pd.Series(np.ones(n)))).rolling(25, min_periods=1).mean().values
    for t in range(n):
        r_v = vol[t] / vol_avg[t] if vol_avg[t] > 0 else 1.0
        beta3[t] = np.clip(beta2[t] * (r_v ** ETA_VOL), 0.2, 5.0)

    lam_vol = np.full(n, l_init)
    bar_beta = np.zeros(n)
    m_pred = np.zeros(n)
    for t in range(n):
        if t:
            if t >= 31:
                err = m_real[t-30:t] - m_pred[t-30:t]
                g = -(2/30)*np.sum(err * m_pred[t-30:t])
                lam_vol[t] = np.clip(lam_vol[t-1] - eta*np.clip(g, -10, 10), l_min, l_max)
            else:
                lam_vol[t] = lam_vol[t-1]
            bar_beta[t] = lam_vol[t]*bar_beta[t-1] + (1-lam_vol[t])*beta3[t]
        else:
            bar_beta[t] = beta3[t]
        m_pred[t] = sig[t] * bar_beta[t]

    diff = m_pred - m_real

    out = pd.DataFrame({
        "Date": df["DispDate"],
        "B_phase1": beta1,
        "B_phase2": beta2,
        "B_phase3": beta3,
        "B_final": bar_beta,
        "M_pred": m_pred,
        "M_real": m_real,
        "M_diff": diff,
        "Norm_err": np.abs(diff) / sig,
        "MAE_5d": pd.Series(diff).abs().rolling(5, min_periods=1).mean(),
        "RelMAE": pd.Series(diff).abs().rolling(5, min_periods=1).mean() / close * 100,
        "HitRate_20d": pd.Series((m_pred >= m_real).astype(int)).rolling(20, min_periods=1).mean() * 100,
    })
    return out

# ──────────────────────────────────────────────────────────────

def make_table(df: pd.DataFrame, title: str = "") -> str:
    dfn = df.tail(NUM_ROWS).iloc[::-1].reset_index(drop=True)

    avg = {"Date": "Average"}
    med = {"Date": "Median"}
    for c in ["B_phase1","B_phase2","B_phase3","B_final","M_pred","M_real","M_diff","Norm_err","MAE_5d","RelMAE","HitRate_20d"]:
        vals = dfn[c].astype(float)
        avg[c] = vals.mean()
        med[c] = np.median(vals)
    dfn = pd.concat([dfn, pd.DataFrame([avg, med])], ignore_index=True)

    cols_src = [
        "Date",
        "B_phase1",
        "B_phase2",
        "B_phase3",
        "B_final",
        "M_pred",
        "M_real",
        "M_diff",
        "Norm_err",
        "MAE_5d",
        "RelMAE",
        "HitRate_20d",
    ]
    header = {
        "B_phase1": r"$\beta^{(1)}$",
        "B_phase2": r"$\beta^{(2)}$",
        "B_phase3": r"$\beta^{(3)}$",
        "B_final": r"$\bar\beta$",
        "M_pred": r"$m_{\mathrm{pred}}$",
        "M_real": r"$m_{\mathrm{real}}$",
        "M_diff": r"$m_\Delta$",
        "Norm_err": r"$|m_\Delta|/\sigma$",
        "MAE_5d": r"$\mathrm{MAE}_5$",
        "RelMAE": r"$\mathrm{RMAE}$",
        "HitRate_20d": r"$\mathrm{HR}_{20}[\%]$",
    }
    cols = [header.get(c, c) for c in cols_src]

    def fmt(v, col):
        if col == "Date":
            return v
        if pd.isna(v):
            return "--"
        if col in {r"$\beta^{(1)}$", r"$\beta^{(2)}$", r"$\beta^{(3)}$", r"$\bar\beta$"}:
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
        fmt_str = "l" + "r"*(len(cols)-1)
        latex_body = disp.to_latex(index=False, escape=False, column_format=fmt_str)

    footnote_lines = [
        r"\begin{tablenotes}\footnotesize",
        r"\item $m_\Delta=m_{\text{pred}}-m_{\text{real}}$,",
        r"$\mathrm{MAE}_5=\mathrm{MAE}_{5\text{d}}$,",
        r"$\mathrm{RMAE}=\mathrm{MAE}_5/\text{Close}$",
        r"\end{tablenotes}",
    ]
    footnote = "\n".join(footnote_lines)

    parts = []
    if title:
        parts.append(rf"\noindent\textbf{{{title}}}\\")
    parts += [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}%",
        r"\begin{threeparttable}",
    ]
    parts += [
        r"\resizebox{\textwidth}{!}{%",
        latex_body.rstrip(),
        r"}",
        footnote,
        r"\end{threeparttable}",
        r"\endgroup",
    ]
    return "\n".join(parts) + "\n"

# ──────────────────────────────────────────────────────────────

def process_one(
    csv: Path,
    out_dir: Path = OUT_DIR,
    *,
    eta: float = ETA,
    l_init: float = L_INIT,
    l_min: float = L_MIN,
    l_max: float = L_MAX,
) -> Path:
    code = csv.stem
    tables: list[str] = []
    for lam, label in VARIANT_LAMBDAS:
        df = calc_range(
            read_prices(csv),
            eta=eta,
            l_init=lam,
            l_min=l_min,
            l_max=l_max,
        )
        title = f"code:{code} λ = {lam:.2f} ({label})"
        tables.append(make_table(df, title))
    tex = "\n\\clearpage\n".join(tables) + "\n"
    out = out_dir / f"{code}_diff.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out

# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="range diff table (30 d ＋ Average)")
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="個別 CSV（省略時は data/prices/*.csv 一括処理）",
    )
    parser.add_argument("--eta", type=float, default=ETA, help="学習率 η")
    parser.add_argument("--init-lambda", type=float, default=L_INIT, help="初期 λ_vol")
    parser.add_argument("--min-lambda", type=float, default=L_MIN, help="最小 λ_vol")
    parser.add_argument("--max-lambda", type=float, default=L_MAX, help="最大 λ_vol")
    args = parser.parse_args()

    kwargs = dict(
        eta=args.eta,
        l_init=args.init_lambda,
        l_min=args.min_lambda,
        l_max=args.max_lambda,
    )

    if args.csv is None:
        for p in sorted(PRICES_DIR.glob("*.csv")):
            out = process_one(p, **kwargs)
            print(f"✅ {p.stem} → {out.relative_to(OUT_DIR.parent.parent)}")
    else:
        out = process_one(resolve_csv(args.csv), **kwargs)
        print(f"✅ {args.csv.stem} → {out.relative_to(OUT_DIR.parent.parent)}")

if __name__ == "__main__":
    main()
