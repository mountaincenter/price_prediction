#!/usr/bin/env python3
"""
scripts/csv_to_open_price_diff.py   v1.6  (2025-06-10)
────────────────────────────────────────────────────────
- CHANGELOG — scripts/csv_to_open_price_diff.py  （newest → oldest）
- 2025-06-10  v1.6 : O_diff/O_real 比率列を追加
- 2025-06-10  v1.5 : 欠損値 '-' を NaN 変換
- 2025-06-10  v1.4 : Phase 5 クリップ (G_fin) 実装
- 2025-06-07  v1.3 : tex-src/open_price の数式を反映し各 Phase を実装
- 2025-06-07  v1.2 : 加法モデルに修正し各 Phase の G 列を追加
- 2025-06-07  v1.1 : Open 用にカラム・計算を修正
- 2025-06-07  v1.0 : 初版（center_shift_diff.py から派生）
"""

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
VARIANT_LAMBDAS = [(0.90, "minimum"), (0.94, "default"), (0.98, "maximum")]
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/analysis/open_price"
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
        "Open": "Open", "始値": "Open",
        "Close":"Close","終値": "Close",
        "5DMA": "5DMA", "５ＤＭＡ": "5DMA",
        "25DVMA": "25DVMA", "２５ＤＶＭＡ": "25DVMA",
        "出来高": "Volume", "Volume": "Volume",
    }
    df = df.rename(columns=rename, errors="ignore")
    if {"Date", "High", "Low", "Open"} - set(df.columns):
        raise KeyError(f"{csv.name}: Date / High / Low / Open が不足")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["DispDate"] = df["Date"].dt.strftime("%m-%d")
    for c in ["High", "Low", "Open", "Close", "Volume", "5DMA", "25DVMA"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .replace({",": "", "-": np.nan}, regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
    df = df.dropna(subset=["High", "Low", "Open"])
    return df.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
def kappa_sigma(s: float) -> float:
    for lo, hi, v in KAPPA_BUCKETS:
        if lo <= s < hi:
            return v
    return 0.20

def calc_open_price(
    df: pd.DataFrame,
    phase: int = 2,
    *,
    eta: float = ETA,
    l_init: float = L_INIT,
    l_min: float = L_MIN,
    l_max: float = L_MAX,
) -> pd.DataFrame:
    n = len(df)
    op = df["Open"].values
    dcl = np.zeros(n); dcl[1:] = np.log(op[1:] / op[:-1])

    sig = np.zeros(n); lam = np.full(n, l_init)
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
            lam[t] = np.clip(lam[t-1] - eta*np.clip(g, -10, 10), l_min, l_max)
        else:
            lam[t] = lam[t-1] if t else l_init

    out = pd.DataFrame({
        "Date": df["DispDate"],
        r"$\kappa(\sigma)$": kap,
        "High": df["High"],
        "Low":  df["Low"],
        r"$\alpha_t$": alp,
        r"$\lambda_{\text{shift}}$": lam,
        r"$\Delta\alpha_t$": dalp,
        r"$\sigma_t^{\mathrm{shift}}$": sig,
        "Open": df["Open"]
    })
    out["B_{t-1}"] = (out["High"].shift(1) + out["Low"].shift(1)) / 2

    gap_raw = out["Open"] - out["B_{t-1}"]
    gap_raw.iloc[0] = 0.0

    g_phase0 = np.zeros(n)
    lam_open = np.full(n, l_init)
    for t in range(n):
        if t:
            g_phase0[t] = lam_open[t-1]*gap_raw.iloc[t] + (1-lam_open[t-1])*g_phase0[t-1]
        else:
            g_phase0[t] = gap_raw.iloc[t]
        if phase >= 4 and t >= 31:
            e = gap_raw.iloc[t-30:t] - g_phase0[t-30:t]
            g = -(2/30) * np.sum(e * g_phase0[t-30:t])
            lam_open[t] = np.clip(lam_open[t-1] - eta*np.clip(g, -10, 10), l_min, l_max)
        else:
            lam_open[t] = lam_open[t-1] if t else l_init

    g_phase1 = g_phase0.copy()
    for t in range(n):
        if t:
            hist = gap_raw.iloc[max(0, t-63):t]
            iqr = np.percentile(hist, 75) - np.percentile(hist, 25)
            if iqr < 1e-4:
                iqr = 1e-4
            s_gap = 1 / iqr
            g_phase1[t] = np.clip(g_phase0[t] * s_gap, -5*sig[t], 5*sig[t])

    g_phase2 = g_phase1.copy()
    sig63 = pd.Series(dcl).rolling(63, min_periods=1).std().values
    for t in range(n):
        r_sig = sig[t] / sig63[t] if sig63[t] > 0 else 1.0
        r_sig = np.clip(r_sig, 0.3, 3.0)
        g_phase2[t] = g_phase1[t] / (r_sig ** 0.5)

    g_final = g_phase2.copy()
    if {"Close", "5DMA", "Volume", "25DVMA"} <= set(df.columns):
        g_proxy = (df["Close"].shift(1) - df["5DMA"].shift(1)) / sig63
        rv = df["Volume"].shift(1) / df["25DVMA"].shift(1)
        wv = np.minimum(1.5, rv.fillna(0.0))
        wv = wv.fillna(0.0)
        g_final = g_phase2 + wv.values * g_proxy.fillna(0.0).values

    g_fin = np.clip(g_final, -5 * sig, 5 * sig)

    out["G_phase0"] = g_phase0
    out["G_phase1"] = g_phase1
    out["G_phase2"] = g_phase2
    out["G_final"] = g_fin

    o_gap = g_fin if phase >= 5 else g_final
    out["O_pred"]  = out["B_{t-1}"] + o_gap
    out["O_real"]  = df["Open"]
    out["O_diff"]  = out["O_pred"] - out["O_real"]
    out["O_ratio"] = np.where(out["O_real"] != 0, out["O_diff"] / out["O_real"], np.nan)

    out["O_diff_sign"] = np.sign(out["O_diff"])
    out["Norm_err"]    = np.abs(out["O_diff"]) / out[r"$\sigma_t^{\mathrm{shift}}$"]
    out["MAE_5d"]      = out["O_diff"].abs().rolling(5, min_periods=1).mean()
    out["RelMAE"]      = out["MAE_5d"] / out["Open"] * 100       # %
    hit = (np.sign(o_gap) ==
           np.sign(out["O_real"] - out["B_{t-1}"])).astype(int)
    out["HitRate_20d"] = hit.rolling(20, min_periods=1).mean() * 100  # %
    return out

# ──────────────────────────────────────────────────────────────
def make_table(df: pd.DataFrame, title: str = "") -> str:
    dfn = df.tail(NUM_ROWS).iloc[::-1].reset_index(drop=True)

    avg = {"Date": "Average"}
    med = {"Date": "Median"}
    for c in [r"$\kappa(\sigma)$","B_{t-1}","G_phase0","G_phase1","G_phase2",
              "G_final","O_pred","O_real","O_diff",
              "O_diff_sign","Norm_err","MAE_5d","RelMAE","HitRate_20d"]:
        vals = dfn[c].astype(float)
        avg[c] = vals.mean()
        med[c] = np.median(vals)
    dfn = pd.concat([dfn, pd.DataFrame([avg, med])], ignore_index=True)

    cols_src = [
        "Date",
        r"$\kappa(\sigma)$",
        "B_{t-1}",
        "G_phase0",
        "G_phase1",
        "G_phase2",
        "G_final",
        "O_pred",
        "O_real",
        "O_diff",
        "O_diff_sign",
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
        "G_phase0":           r"$G_{\mathrm{ew}}$",
        "G_phase1":           r"$G_{\mathrm{sc}}$",
        "G_phase2":           r"$G_{\mathrm{vol}}$",
        "G_final":            r"$G_{\mathrm{fin}}$",
        "O_pred":             r"$O_p$",
        "O_real":             r"$O_r$",
        "O_diff":             r"$O_\Delta$",
        "O_diff_sign":        r"$\mathrm{sgn}\,O_\Delta$",
        "Norm_err":           r"$|O_\Delta|/\sigma$",
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
        r"$O_p=O_{\text{pred}}$, $O_r=O_{\text{real}}$, "
        r"$O_\Delta=O_{\text{diff}}$, "
        r"$\mathrm{sgn}\,O_\Delta=\operatorname{sign}(O_{\text{diff}})$, "
        r"$|O_\Delta|/\sigma=\dfrac{|O_{\text{diff}}|}{\sigma_t^{\text{shift}}}$, "
        r"$\mathrm{MAE}_5=\mathrm{MAE}_{5\text{d}}$, "
        r"$\mathrm{RMAE}= \mathrm{MAE}_5 / \text{Open}$, "
        r"$\mathrm{HR}_{20}=\mathrm{HitRate}_{20\text{d}}$, ",
        r"$\lambda_{\text{shift}}=\lambda_t$, ",
        r"$\Delta\alpha_t=\alpha_t-\alpha_{t-1}$.",
        r"\end{tablenotes}"
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
        r"\endgroup"
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
    """csv を処理して diff.tex を生成し、そのパスを返す"""
    code = csv.stem
    tables: list[str] = []
    for lam, label in VARIANT_LAMBDAS:
        df = calc_open_price(
            read_prices(csv),
            phase=5,
            eta=eta,
            l_init=lam,
            l_min=lam,
            l_max=lam,
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
    parser = argparse.ArgumentParser(
        description="open_price diff table (30 d ＋ Average)"
    )
    parser.add_argument(
        "csv", nargs="?", type=Path,
        help="個別 CSV（省略時は data/prices/*.csv 一括処理）",
    )
    parser.add_argument("--eta", type=float, default=ETA, help="学習率 η")
    parser.add_argument("--init-lambda", type=float, default=L_INIT, help="初期 λ_shift")
    parser.add_argument("--min-lambda", type=float, default=L_MIN, help="最小 λ_shift")
    parser.add_argument("--max-lambda", type=float, default=L_MAX, help="最大 λ_shift")
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
