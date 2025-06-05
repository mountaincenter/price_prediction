#!/usr/bin/env python3
"""
scripts/csv_to_basic_form_tex.py   v1.7  (2025-06-03)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-03  v1.7 : 予測列の shift(-1) を撤回し当日行に対応付け
- 2025-06-03  v1.6 : 最新日から 14 行抽出し降順表示
- 2025-06-03  v1.5 : Date を YYYY-MM-DD / float_format %.1f
- 2025-06-03  v1.4 : \resizebox 追加・Date 昇順・予測列 1 日シフト
- 2025-06-03  v1.3 : \scriptsize ラッパでフォント縮小
- 2025-06-03  v1.2 : 価格列カンマ除去・空白トリム対応
- 2025-06-03  v1.1 : 価格列を数値型へ強制変換
- 2025-06-03  v1.0 : 初版 — basic_form Phase-0 予測付き .tex 生成
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────────────────────
def _resolve_csv_path(raw: Path) -> Path:
    if raw.exists():
        return raw.resolve()
    alt = Path(__file__).resolve().parent.parent.parent / "data" / "prices" / raw.name
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"CSV が見つかりません: {raw} または {alt}")


def _preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    # 1. 列リネーム
    df_raw.columns = df_raw.columns.str.strip().str.replace("　", "")
    rename = {
        "Date": "Date", "date": "Date", "日付": "Date",
        "Open": "O_t",  "open": "O_t",  "始値": "O_t",
        "High": "H_t",  "high": "H_t",  "高値": "H_t",
        "Low":  "L_t",  "low": "L_t",  "安値": "L_t",
        "Close": "Cl_t","close": "Cl_t","終値": "Cl_t",
    }
    df = df_raw.rename(columns=rename, errors="ignore")

    req = ["Date", "O_t", "H_t", "L_t", "Cl_t"]
    if missing := [c for c in req if c not in df.columns]:
        raise KeyError(f"必須列が CSV に存在しません: {missing}")

    # 2. Date 整形・昇順
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)

    # 3. 価格列数値化
    price_cols = ["O_t", "H_t", "L_t", "Cl_t"]
    df[price_cols] = (
        df[price_cols]
        .replace({",": ""}, regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=price_cols, how="all")

    # 4. 実績 σ_t & Cl_{t-1}
    df["sigma_t"] = df["H_t"] - df["L_t"]
    df["Cl_{t-1}"] = df["Cl_t"].shift(1)

    # 5. Phase-0 予測（前日データに基づき **当日** を予測）
    pred_sigma = df["sigma_t"].shift(1)
    pred_base  = df["Cl_{t-1}"]
    df["sigma_t_pred"] = pred_sigma
    df["O_t_pred"]  = pred_base
    df["H_t_pred"]  = pred_base + pred_sigma
    df["L_t_pred"]  = pred_base - pred_sigma
    df["Cl_t_pred"] = pred_base
    # ※ v1.4 の shift(-1) を撤回

    # 6. 列順 & 数式ヘッダ
    df = df[
        ["Date","O_t","H_t","L_t","Cl_t",
         "Cl_{t-1}","sigma_t","sigma_t_pred",
         "O_t_pred","H_t_pred","L_t_pred","Cl_t_pred"]
    ]
    display = {
        "Date": "Date",
        "O_t": r"$O_{t}$",  "H_t": r"$H_{t}$",  "L_t": r"$L_{t}$",
        "Cl_t": r"$Cl_{t}$", "Cl_{t-1}": r"$Cl_{t-1}$",
        "sigma_t": r"$\sigma_{t}$",
        "sigma_t_pred": r"$\sigma_{t}^{\mathrm{pred}}$",
        "O_t_pred": r"$O_{t}^{\mathrm{pred}}$",
        "H_t_pred": r"$H_{t}^{\mathrm{pred}}$",
        "L_t_pred": r"$L_{t}^{\mathrm{pred}}$",
        "Cl_t_pred": r"$Cl_{t}^{\mathrm{pred}}$",
    }
    df = df.rename(columns=display)

    # 7. 最新 14 行 → 降順表示
    df_last14 = df.tail(14).iloc[::-1].reset_index(drop=True)
    return df_last14


def _df_to_tex(df: pd.DataFrame) -> str:
    tabular = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        float_format="%.1f",
    )
    return (
        r"\begingroup"                     "\n"
        r"\scriptsize"                     "\n"
        r"\resizebox{\textwidth}{!}{%"     "\n"
        f"{tabular.strip()}"               "\n"
        r"}%"                              "\n"
        r"\endgroup"                       "\n"
    )


# ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="basic_form Phase-0 予測付き LaTeX スニペット生成"
    )
    p.add_argument("csv", type=Path, help="data/prices/<code>.csv")
    p.add_argument("--out", "-o", type=Path, help="出力パスを明示指定")
    args = p.parse_args()

    csv_path = _resolve_csv_path(args.csv)
    code = csv_path.stem

    out_path = (
        args.out.resolve()
        if args.out
        else (
            Path(__file__).resolve().parent.parent.parent
            / "data" / "analysis" / "basic_form" / f"{code}.tex"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _preprocess(pd.read_csv(csv_path))
    out_path.write_text(_df_to_tex(df), encoding="utf-8")
    print(f"✅ 生成完了: {out_path}")


if __name__ == "__main__":
    main()
