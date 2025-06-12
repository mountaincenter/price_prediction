#!/usr/bin/env python3
"""scripts/csv_to_integrate.py   v1.1  (2025-06-12)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-12  v1.1 : 比率列を百分率(小数第1位)表示
- 2025-06-12  v1.0 : 初版
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from csv_to_open_price_diff import read_prices as read_prices, calc_open_price
from csv_to_event_diff import calc_event_beta
from csv_to_range_diff import calc_range

NUM_ROWS = 30
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/integrate"
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/prices"

# ---------------------------------------------------------------------------
def resolve_csv(raw: Path) -> Path:
    if raw.exists():
        return raw.resolve()
    alt = PRICES_DIR / raw.name
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(raw)

# ---------------------------------------------------------------------------
def calc_integrate(df: pd.DataFrame, *, code: str) -> pd.DataFrame:
    df_open = calc_open_price(df.copy(), phase=5)
    df_event = calc_event_beta(df.copy(), code=code)
    df_range = calc_range(df.copy())

    cl_p = df_event["C_pred_evt"]
    m_fin = df_range["M_final"]

    out = pd.DataFrame({
        "Date": df_open["Date"],
        "O_p": df_open["O_pred"],
        "O_r": df_open["O_real"],
    })
    out["O_diff"] = out["O_p"] - out["O_r"]
    out["O_diff/O_r"] = np.where(out["O_r"] != 0, out["O_diff"] / out["O_r"], np.nan)

    out["H_p"] = cl_p + m_fin
    out["H_r"] = df_open["High"]
    out["H_diff"] = out["H_p"] - out["H_r"]
    out["H_diff/H_r"] = np.where(out["H_r"] != 0, out["H_diff"] / out["H_r"], np.nan)

    out["L_p"] = cl_p - m_fin
    out["L_r"] = df_open["Low"]
    out["L_diff"] = out["L_p"] - out["L_r"]
    out["L_diff/Lr"] = np.where(out["L_r"] != 0, out["L_diff"] / out["L_r"], np.nan)

    out["Cl_p"] = cl_p
    out["Cl_r"] = df_event["Close"]
    out["Cl_diff"] = out["Cl_p"] - out["Cl_r"]
    out["Cl_diff/Cl_r"] = np.where(out["Cl_r"] != 0, out["Cl_diff"] / out["Cl_r"], np.nan)

    return out

# ---------------------------------------------------------------------------
def make_table(df: pd.DataFrame, title: str = "") -> str:
    dfn = df.tail(NUM_ROWS).iloc[::-1].reset_index(drop=True)

    avg = {"Date": "Average"}
    med = {"Date": "Median"}
    cols_calc = [
        "O_p","O_r","O_diff","O_diff/O_r",
        "H_p","H_r","H_diff","H_diff/H_r",
        "L_p","L_r","L_diff","L_diff/Lr",
        "Cl_p","Cl_r","Cl_diff","Cl_diff/Cl_r",
    ]
    for c in cols_calc:
        vals = dfn[c].astype(float)
        avg[c] = vals.mean()
        med[c] = np.median(vals)
    dfn = pd.concat([dfn, pd.DataFrame([avg, med])], ignore_index=True)

    cols_src = [
        "Date","O_p","O_r","O_diff","O_diff/O_r",
        "H_p","H_r","H_diff","H_diff/H_r",
        "L_p","L_r","L_diff","L_diff/Lr",
        "Cl_p","Cl_r","Cl_diff","Cl_diff/Cl_r",
    ]
    header = {
        "O_p": r"$O_p$",
        "O_r": r"$O_r$",
        "O_diff": r"$O_\Delta$",
        "O_diff/O_r": r"$O_\Delta/O_r$",
        "H_p": r"$H_p$",
        "H_r": r"$H_r$",
        "H_diff": r"$H_\Delta$",
        "H_diff/H_r": r"$H_\Delta/H_r$",
        "L_p": r"$L_p$",
        "L_r": r"$L_r$",
        "L_diff": r"$L_\Delta$",
        "L_diff/Lr": r"$L_\Delta/L_r$",
        "Cl_p": r"$Cl_p$",
        "Cl_r": r"$Cl_r$",
        "Cl_diff": r"$Cl_\Delta$",
        "Cl_diff/Cl_r": r"$Cl_\Delta/Cl_r$",
    }
    cols = [header.get(c, c) for c in cols_src]

    def fmt(v, col):
        if col == "Date":
            return v
        if pd.isna(v):
            return "--"
        if col in {r"$O_\Delta/O_r$", r"$H_\Delta/H_r$", r"$L_\Delta/L_r$", r"$Cl_\Delta/Cl_r$"}:
            return f"{100 * v:.1f}"
        return f"{v:.1f}"

    disp = pd.DataFrame({
        cols[i]: [fmt(v, cols[i]) for v in dfn[cols_src[i]]]
        for i in range(len(cols))
    })

    fmt_str = "l" + "r" * (len(cols) - 1)
    latex_body = disp.to_latex(index=False, escape=False, column_format=fmt_str)

    footnote = "\n".join([
        r"\begin{tablenotes}\footnotesize",
        r"\item $O_p=O_{\text{pred}}$, $O_r=O_{\text{real}}$, $O_\Delta=O_{\text{diff}}$.",
        r"\item $H_p=H_{\text{pred}}$, $L_p=L_{\text{pred}}$, $Cl_p=C_{\text{pred}}$.",
        r"\item $O_\Delta/O_r$ などの比率は \(\times100\) で示す。",
        r"\end{tablenotes}",
    ])

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

# ---------------------------------------------------------------------------
def process_one(csv: Path, out_dir: Path = OUT_DIR) -> Path:
    code = csv.stem
    df = calc_integrate(read_prices(csv), code=code)
    tex = make_table(df, title=f"code:{code}")
    out = out_dir / f"{code}_diff.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding='utf-8')
    return out

# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="integrated diff table")
    parser.add_argument("csv", nargs="?", type=Path, help="個別 CSV（省略時は data/prices/*.csv 一括処理）")
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
