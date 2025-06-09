#!/usr/bin/env python3
"""scripts/csv_to_event_diff.py   v1.4  (2025-06-11)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-11  v1.4 : β 各要素の中間変数を出力
- 2025-06-09  v1.3 : 出力カラムを整理し \u03b1/\u03bb 関連列を削除
- 2025-06-10  v1.2 : \u03b2_earn を指数平滑化
- 2025-06-10  v1.1 : weekday/earn/market 各フェーズの簡易実装を追加
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
EARN_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/earn"
MARKET_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data/market"

# ──────────────────────────────────────────────────────────────
def resolve_csv(raw: Path) -> Path:
    if raw.exists():
        return raw.resolve()
    alt = PRICES_DIR / raw.name
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(raw)

# ──────────────────────────────────────────────────────────────
def load_index(name: str) -> pd.DataFrame:
    path = MARKET_DIR / f"{name}.csv"
    df = pd.read_csv(path).rename(columns=lambda c: c.strip().replace("　", ""))
    rename = {"Date": "Date", "日付": "Date", "Close": "Close", "終値": "Close"}
    df = df.rename(columns=rename, errors="ignore")
    if "Close" in df.columns:
        df["Close"] = df["Close"].replace({",": ""}, regex=True).astype(float)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret"] = np.log(df["Close"]).diff()
    df["std"] = df["ret"].rolling(63, min_periods=2).std()
    return df[["Date", "ret", "std"]]


def load_earn_dates(code: str) -> set[pd.Timestamp]:
    path = EARN_DIR / f"{code}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["DisclosedDate"], dtype=str)
    return set(pd.to_datetime(df["DisclosedDate"], errors="coerce").dropna().dt.normalize())


def compute_beta_weekday(dates: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(dates)
    wk_map = {0: 1.10, 1: 1.05, 2: 1.00, 3: 0.98, 4: 0.95}
    base = dates.dt.dayofweek.map(wk_map).fillna(1.0).values

    diff_next = (dates.shift(-1) - dates).dt.days.fillna(1)
    diff_prev = (dates - dates.shift(1)).dt.days.fillna(1)
    holiday = np.ones(n)
    holiday[diff_next > 1] *= 0.90
    holiday[diff_prev > 1] *= 0.95
    inp = base * holiday

    lam = 0.90
    out = np.empty(n)
    for t in range(n):
        out[t] = inp[t] if t == 0 else lam * out[t-1] + (1 - lam) * inp[t]
        out[t] = np.clip(out[t], 0.8, 1.2)
    return out, base, holiday


def compute_beta_earn(dates: pd.Series, earn_dates: set[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
    n = len(dates)
    raw = np.ones(n)
    date_idx = {d.normalize(): i for i, d in enumerate(dates)}
    for d in earn_dates:
        if d in date_idx:
            i = date_idx[d]
        else:
            i = np.searchsorted(dates.values, np.datetime64(d))
            if i >= n:
                continue
        raw[i] = max(raw[i], 1.20)
        if i > 0:
            raw[i-1] = max(raw[i-1], 1.15)
        if i + 1 < n:
            raw[i+1] = max(raw[i+1], 1.10)
    lam = 0.80
    beta = np.empty(n)
    for t in range(n):
        beta[t] = raw[t] if t == 0 else lam * beta[t-1] + (1 - lam) * raw[t]
        beta[t] = np.clip(beta[t], 0.8, 1.5)
    return beta, raw


def compute_beta_market(
    dates: pd.Series,
    close: pd.Series,
    idx: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ret_code = np.log(close).diff()
    df_code = pd.DataFrame({"Date": dates, "ret_code": ret_code})
    prod = np.ones(len(dates))
    lam = 0.90
    comp: dict[str, np.ndarray] = {}
    for name, dfi in idx.items():
        merged = pd.merge(df_code, dfi, on="Date", how="left").fillna(method="ffill")
        rho = merged["ret_code"].rolling(63, min_periods=20).corr(merged["ret"])
        z = merged["ret"] / merged["std"]
        beta = (1 + rho * z).clip(0.8, 1.2).fillna(1.0)
        ew = np.empty(len(beta))
        for t in range(len(beta)):
            ew[t] = beta.iloc[t] if t == 0 else lam * ew[t-1] + (1 - lam) * beta.iloc[t]
            ew[t] = np.clip(ew[t], 0.8, 1.2)
        prod *= ew
        comp[name] = ew
    return prod, comp

# ──────────────────────────────────────────────────────────────
def calc_event_beta(
    df: pd.DataFrame,
    code: str | None = None,
    *,
    eta: float = ETA,
    l_init: float = L_INIT,
    l_min: float = L_MIN,
    l_max: float = L_MAX,
) -> pd.DataFrame:
    base = calc_center_shift(df, phase=2, eta=eta, l_init=l_init, l_min=l_min, l_max=l_max)

    beta_weekday, wd_base, wd_holiday = compute_beta_weekday(df["Date"])
    earn_dates = load_earn_dates(code) if code else set()
    beta_earn, earn_raw = compute_beta_earn(df["Date"], earn_dates)
    idx = {
        "topix": load_index("topix"),
        "sp500": load_index("sp500"),
        "usd_jpy": load_index("usd_jpy"),
    }
    beta_market, beta_comp = compute_beta_market(df["Date"], df["Close"], idx)
    beta_event = beta_weekday * beta_earn * beta_market

    out = base.copy()
    out["Beta_wd_base"] = wd_base
    out["Beta_wd_holiday"] = wd_holiday
    out["Beta_weekday"] = beta_weekday
    out["Beta_earn_raw"] = earn_raw
    out["Beta_earn"] = beta_earn
    for name, arr in beta_comp.items():
        out[f"Beta_mkt_{name}"] = arr
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
        "Beta_wd_base","Beta_wd_holiday","Beta_weekday",
        "Beta_earn_raw","Beta_earn",
        "Beta_mkt_topix","Beta_mkt_sp500","Beta_mkt_usd_jpy","Beta_market",
        "Beta_event",
        "B_{t-1}","C_pred_evt","C_real","C_diff","C_diff_sign","Norm_err",
    ]:
        vals = dfn[c].astype(float)
        avg[c] = vals.mean()
        med[c] = np.median(vals)
    dfn = pd.concat([dfn, pd.DataFrame([avg, med])], ignore_index=True)

    cols_src = [
        "Date",
        "Beta_wd_base","Beta_wd_holiday","Beta_weekday",
        "Beta_earn_raw","Beta_earn",
        "Beta_mkt_topix","Beta_mkt_sp500","Beta_mkt_usd_jpy","Beta_market",
        "Beta_event",
        "B_{t-1}","C_pred_evt","C_real","C_diff","C_diff_sign","Norm_err",
    ]
    header = {
        "Beta_wd_base": r"$wd_b$",
        "Beta_wd_holiday": r"$wd_h$",
        "Beta_weekday": r"$\beta_{wd}$",
        "Beta_earn_raw": r"$earn_r$",
        "Beta_earn": r"$\beta_{earn}$",
        "Beta_mkt_topix": r"$mkt_{TPX}$",
        "Beta_mkt_sp500": r"$mkt_{SP5}$",
        "Beta_mkt_usd_jpy": r"$mkt_{USD}$",
        "Beta_market": r"$\beta_{mkt}$",
        "Beta_event": r"$\beta_{evt}$",
        "B_{t-1}": r"$B$",
        "C_pred_evt": r"$C_p$",
        "C_real": r"$C_r$",
        "C_diff": r"$C_\Delta$",
        "C_diff_sign": r"$\mathrm{sgn}\,C_\Delta$",
        "Norm_err": r"$|C_\Delta|/\sigma$",
    }
    cols = [header.get(c, c) for c in cols_src]

    def fmt(v, col):
        if col == "Date":
            return v
        if pd.isna(v):
            return "--"
        if col in {
            r"$\beta_{wd}$", r"$\beta_{earn}$", r"$\beta_{mkt}$", r"$\beta_{evt}$",
            r"$wd_b$", r"$wd_h$", r"$earn_r$",
            r"$mkt_{TPX}$", r"$mkt_{SP5}$", r"$mkt_{USD}$",
        }:
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
        r"\item $wd_b$=weekday base, $wd_h$=holiday adj.,",
        r"$earn_r$=raw earn beta, $mkt_{*}$=market beta comps,",
        r"$\beta_{wd}=\beta_{weekday}$, $B=B_{t-1}$,",
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
        code=code,
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
