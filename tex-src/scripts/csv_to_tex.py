#!/usr/bin/env python3
"""
scripts/csv_to_tex.py   v6.1  (2025-06-03)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-05  v6.1 : ルート基準でパスを解決
- 2025-06-03  日付列を先頭に保持（Date → 最初の列に追加）                ← v6.0
- 2025-06-03  列ヘッダを数式モード下付き表記 ($O_{t}$ 等) に変更            ← v5.0
- 2025-06-03  O_t/H_t/L_t/Cl_t/Cl_{t-1} 5 列だけ出力 & カラム名変換対応     ← v4.0
- 2025-06-03  スニペット方式へ切替（header／footer を出力しない）          ← v3.0
- 2025-06-03  data/prices パス自動探索ロジックを追加                         ← v2.0
- 2025-06-03  初回作成: CSV→LaTeX 変換ツール                                 ← v1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_csv_path(raw_path: Path) -> Path:
    """
    引数で与えられた CSV パスを解決する。

    1. そのまま存在するか確認
    2. プロジェクトルート直下の data/prices/ に同名ファイルがあれば採用
    """
    if raw_path.exists():
        return raw_path.resolve()

    project_root = Path(__file__).resolve().parent.parent.parent
    alt_path = project_root / "tex-src" / "data" / "prices" / raw_path.name
    if alt_path.exists():
        return alt_path.resolve()

    raise FileNotFoundError(f"CSV が見つかりません: {raw_path} または {alt_path}")


def _rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名を変換し、必要列を抽出。

    - Date 列（Date/date/日付）を先頭に保持
    - Open   → O_t
    - High   → H_t
    - Low    → L_t
    - Close  → Cl_t
    - Cl_{t-1} = Cl_t.shift(1)

    列ヘッダは数式モード下付き表記（Date はそのまま）。
    """

    # 日付列名判定
    date_candidates = ["Date", "date", "日付"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise KeyError("CSV に日付列 (Date/date/日付) が見つかりません")

    # 価格列リネーム
    rename_map = {
        "Open": "O_t",
        "open": "O_t",
        "始値": "O_t",
        "High": "H_t",
        "high": "H_t",
        "高値": "H_t",
        "Low": "L_t",
        "low": "L_t",
        "安値": "L_t",
        "Close": "Cl_t",
        "close": "Cl_t",
        "終値": "Cl_t",
    }
    df = df.rename(columns=rename_map, errors="ignore")

    # 必要列チェック
    required_cols = ["O_t", "H_t", "L_t", "Cl_t"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV に必要列が見つかりません: {missing}")

    # 前日終値
    df["Cl_{t-1}"] = df["Cl_t"].shift(1)

    # 列順序
    df = df[[date_col, "O_t", "H_t", "L_t", "Cl_t", "Cl_{t-1}"]]

    # 数式モード表記
    display_map = {
        "O_t": r"$O_{t}$",
        "H_t": r"$H_{t}$",
        "L_t": r"$L_{t}$",
        "Cl_t": r"$Cl_{t}$",
        "Cl_{t-1}": r"$Cl_{t-1}$",
    }
    df = df.rename(columns=display_map)

    # 日付列はヘッダ "Date" に統一
    df = df.rename(columns={date_col: "Date"})

    return df


def csv_to_tex(input_path: Path, output_path: Path) -> None:
    """
    与えられた CSV を LaTeX tabular スニペットとして書き出す。
    """
    df_raw = pd.read_csv(input_path)
    df = _rename_and_select(df_raw)

    tabular = df.to_latex(
        index=False,
        escape=False,  # 数式モード保持
        longtable=False,
        caption=(
            f"{input_path.stem} 価格データ "
            "(Date, $O_{t}$, $H_{t}$, $L_{t}$, $Cl_{t}$, $Cl_{t-1}$)"
        ),
        label=f"tbl:{input_path.stem}",
        float_format="%.4f",
    )

    output_path.write_text(tabular, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV を LaTeX tabular スニペットへ変換するスクリプト"
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="変換対象の CSV ファイル（例: 8801.csv）",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="出力 .tex ファイル（省略時は <CSV 名>.tex）",
    )

    args = parser.parse_args()

    csv_path: Path = resolve_csv_path(args.csv)
    tex_path: Path = (
        args.out.resolve()
        if args.out is not None
        else csv_path.with_suffix(".tex").resolve()
    )

    csv_to_tex(csv_path, tex_path)
    print(f"✅ 生成完了: {tex_path}")


if __name__ == "__main__":
    main()
