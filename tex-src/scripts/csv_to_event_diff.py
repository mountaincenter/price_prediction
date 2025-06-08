#!/usr/bin/env python3
"""scripts/csv_to_event_diff.py   v1.0  (2025-06-10)
────────────────────────────────────────────────────────
- CHANGELOG — scripts/csv_to_event_diff.py  （newest → oldest）
- 2025-06-10  v1.0 : 初版
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# center_shift のロジックを流用
from csv_to_center_shift_diff import (
    read_prices,
    calc_center_shift,
    ETA,
    L_INIT,
    L_MIN,
    L_MAX,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/event"
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/prices"

# ──────────────────────────────────────────────────────────────
def resolve_csv(raw: Path) -> Path:
    if raw.exists():
        return raw.resolve()
    alt = PRICES_DIR / raw.name
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(raw)

# ──────────────────────────────────────────────────────────────
def calc_event_beta(
    df: pd.DataFrame,
    *,
    eta: float = ETA,
    l_init: float = L_INIT,
    l_min: float = L_MIN,
    l_max: float = L_MAX,
) -> pd.DataFrame:
    base = calc_center_shift(df, phase=2, eta=eta, l_init=l_init, l_min=l_min, l_max=l_max)

    dow = df["Date"].dt.dayofweek
    wk_map = {0: 1.05, 1: 1.02, 2: 1.00, 3: 0.98, 4: 0.95}
    beta_weekday = dow.map(wk_map).fillna(1.0)
    beta_earn = np.ones(len(df))
    beta_market = np.ones(len(df))
    beta_event = beta_weekday * beta_earn * beta_market

    out = base.copy()
    out["Beta_weekday"] = beta_weekday.values
    out["Beta_earn"] = beta_earn
    out["Beta_market"] = beta_market
    out["Beta_event"] = beta_event

    out["C_pred_evt"] = out["C_pred"] * out["Beta_event"]
    out["C_diff"] = out["C_pred_evt"] - out["C_real"]
    out["C_diff_sign"] = np.sign(out["C_diff"])
    out["Norm_err"] = np.abs(out["C_diff"]) / (out["B_{t-1}"] * out[r"$\sigma_t^{\mathrm{shift}}$"])
    out["MAE_5d"] = out["C_diff"].abs().rolling(5, min_periods=1).mean()
    out["RelMAE"] = out["MAE_5d"] / out["Close"] * 100
    hit = (np.sign(out[r"$\alpha_t$"]) == np.sign(out["C_real"] - out["B_{t-1}"])).astype(int)
    out["HitRate_20d"] = hit.rolling(20, min_periods=1).mean() * 100
    return out

# ──────────────────────────────────────────────────────────────
NUM_ROWS = 30

def make_table(df: pd.DataFrame, title: str = "") -> str:
    dfn = df.tail(NUM_ROWS).iloc[::-1].reset_index(drop=True)

    avg = {"Date": "Average"}
    med = {"Date": "Median"}
    for c in [
        "Beta_weekday","Beta_earn","Beta_market","Beta_event",
        "B_{t-1}","C_pred_evt","C_real","C_diff","C_diff_sign","Norm_err",
        "MAE_5d","RelMAE","HitRate_20d",
    ]:
        vals = dfn[c].astype(float)
        avg[c] = vals.mean()
        med[c] = np.median(vals)
    dfn = pd.concat([dfn, pd.DataFrame([avg, med])], ignore_index=True)

    cols_src = [
        "Date","Beta_weekday","Beta_earn","Beta_market","Beta_event",
        "B_{t-1}","C_pred_evt","C_real","C_diff","C_diff_sign","Norm_err",
        r"$\alpha_t$",r"$\lambda_{\text{shift}}$",r"$\Delta\alpha_t$",
        "MAE_5d","RelMAE","HitRate_20d",
    ]
    header = {
        "Beta_weekday": r"$\beta_{wd}$",
        "Beta_earn": r"$\beta_{earn}$",
        "Beta_market": r"$\beta_{mkt}$",
        "Beta_event": r"$\beta_{evt}$",
        "B_{t-1}": r"$B$",
        "C_pred_evt": r"$C_p$",
        "C_real": r"$C_r$",
        "C_diff": r"$C_\Delta$",
        "C_diff_sign": r"$\mathrm{sgn}\,C_\Delta$",
        "Norm_err": r"$|C_\Delta|/\sigma$",
        r"$\alpha_t$": r"$\alpha_t$",
        r"$\lambda_{\text{shift}}$": r"$\lambda$",
        r"$\Delta\alpha_t$": r"$\Delta\alpha$",
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
        if col in {r"$\beta_{wd}$", r"$\beta_{earn}$", r"$\beta_{mkt}$", r"$\beta_{evt}$",
                    r"$\alpha_t$", r"$\lambda$", r"$\Delta\alpha$"}:
            return f"{v:.2f}"
        if col in {r"$\mathrm{RMAE}$", r"$\mathrm{HR}_{20}[\%]$"}:
            return f"{v:.2f}"
        return f"{v:.1f}"

    disp = pd.DataFrame({
        cols[i]: [fmt(v, cols[i]) for v in dfn[cols_src[i]]]
        for i in range(len(cols))
    })

    with pd.option_context("mode.chained_assignment", None):
        fmt_str = "l" + "r" * (len(cols) - 1)
        latex_body = disp.to_latex(index=False, escape=False, column_format=fmt_str)

    footnote_lines = [
        r"\begin{tablenotes}\footnotesize",
        r"\item $\beta_{wd}=\beta_{weekday}$, $B=B_{t-1}$,",
        r"$C_p=C_{\text{pred}}\beta_{evt}$, $C_r=C_{\text{real}}$,",
        r"$C_\Delta=C_{\text{diff}}$, $|C_\Delta|/\sigma=|C_{\text{diff}}|/\sigma_t^{\text{shift}}$.",
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
    df = calc_event_beta(
        read_prices(csv),
        eta=eta,
        l_init=l_init,
        l_min=l_min,
        l_max=l_max,
    )
    tex = make_table(df, title=f"code:{code}")
    out = out_dir / f"{code}_diff.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    return out

# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="event diff table")
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="個別 CSV（省略時は data/prices/*.csv 一括処理）",
    )
    parser.add_argument("--eta", type=float, default=ETA, help="学習率 η")
    parser.add_argument("--init-lambda", type=float, default=L_INIT, help="初期 λ_shift")
    parser.add_argument("--min-lambda", type=float, default=L_MIN, help="最小 λ_shift")
    parser.add_argument("--max-lambda", type=float, default=L_MAX, help="最大 λ_shift")
    args = parser.parse_args()

    kwargs = dict(eta=args.eta, l_init=args.init_lambda, l_min=args.min_lambda, l_max=args.max_lambda)

    if args.csv is None:
        for p in sorted(PRICES_DIR.glob("*.csv")):
            out = process_one(p, **kwargs)
            print(f"✅ {p.stem} → {out.relative_to(OUT_DIR.parent.parent)}")
    else:
        out = process_one(resolve_csv(args.csv), **kwargs)
        print(f"✅ {args.csv.stem} → {out.relative_to(OUT_DIR.parent.parent)}")

if __name__ == "__main__":
    main()
