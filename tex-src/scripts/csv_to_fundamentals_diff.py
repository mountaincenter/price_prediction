#!/usr/bin/env python3
"""scripts/csv_to_fundamentals_diff.py
  v1.5  (2025-06-13)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-13  v1.5 : Outlier=3 を保持可能にし Total は非ゼロ件数を計上
- 2025-06-12  v1.4 : center_shift がイベント日判定 (2) に対応
- 2025-06-13  v1.3 : Total 列を追加し長大表を longtable で出力、CSV も保存
- 2025-06-12  Row ラベル中の `_` を `\_` にエスケープして LaTeX コンパイルエラーを解消
- 2025-06-10  v1.1 : 単独コンパイル可能な TeX を出力
- 2025-06-10  v1.0 : 初版
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

# 既存モジュール再利用
from csv_to_center_shift_diff import (
    read_prices,
    calc_center_shift,
)

START_DATE = pd.Timestamp("2024-01-04")
END_DATE   = pd.Timestamp("2025-06-09")

PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data" / "prices"
EVENTS_CSV = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data" / "events.csv"
OUT_TEX    = Path(__file__).resolve().parent.parent.parent / "tex-src" / "fundamentals.tex"
OUT_CSV    = OUT_TEX.with_suffix('.csv')

# ──────────────────────────────────────────────────────────────
def read_events(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["Date"] = pd.to_datetime(df["Date"].str.replace("(予定)", "", regex=False))
    df["Code"] = df["国"].map({"日本": "J", "中国": "C", "米国": "U"})
    df["IsTrump"] = df["カテゴリ"].str.contains("トランプ")
    return df[["Date", "Code", "IsTrump"]]

# ──────────────────────────────────────────────────────────────
def collect_outliers(prices_dir: Path) -> tuple[list[str], dict[str, dict[pd.Timestamp, int]]]:
    codes: list[str] = []
    data: dict[str, dict[pd.Timestamp, int]] = {}
    for csv_path in sorted(prices_dir.glob("*.csv")):
        code = csv_path.stem
        raw = read_prices(csv_path)
        df = calc_center_shift(raw, events_csv=EVENTS_CSV)
        df["Date_full"] = raw["Date"]
        mask = (df["Date_full"] >= START_DATE) & (df["Date_full"] <= END_DATE)
        flags = {
            d.date(): int(o) for d, o in zip(df.loc[mask, "Date_full"], df.loc[mask, "Outlier"])
        }
        codes.append(code)
        data[code] = flags
    return codes, data

# ──────────────────────────────────────────────────────────────
def build_rows(
    codes: list[str],
    data: dict[str, dict[pd.Timestamp, int]],
    events: pd.DataFrame,
) -> list[dict[str, int]]:
    all_dates = sorted({d for flags in data.values() for d in flags})
    rows: list[dict[str, int]] = []
    for d in all_dates:
        base = {c: data[c].get(d, 0) for c in codes}
        total = sum(1 for v in base.values() if v != 0)
        evs: list[str] = []
        for _, r in events.iterrows():
            delta = (d - r["Date"].date()).days
            if r["IsTrump"]:
                if delta in {0, 1}:
                    tag = f"{d}_{'T'}_d" + ("" if delta == 0 else "+1")
                    evs.append(tag)
            else:
                if delta in {-1, 0, 1}:
                    sign = "" if delta == 0 else ("+1" if delta == 1 else "-1")
                    tag = f"{d}_{r['Code']}_d{sign}"
                    evs.append(tag)
        if not evs:
            rows.append({"Row": str(d), **base, "Total": total})
        else:
            for t in evs:
                rows.append({"Row": t, **base, "Total": total})
    return rows

# ──────────────────────────────────────────────────────────────
def make_table(rows: list[dict[str, int]], codes: list[str]) -> str:
    """DataFrame → LaTeX 変換（Row ラベルの _ をエスケープ）"""
    df = pd.DataFrame(rows)
    df = df[["Row", *codes, "Total"]]
    # Row 列のアンダースコアを LaTeX 用にエスケープ
    df["Row"] = df["Row"].str.replace("_", r"\_", regex=False)
    fmt_str = "l" + "c" * (len(codes) + 1)
    latex = df.to_latex(index=False, escape=False, column_format=fmt_str, longtable=True)
    return "\n".join(
        [
            r"\begingroup",
            r"\footnotesize",
            latex.rstrip(),
            r"\endgroup",
            "",
        ]
    )

# ──────────────────────────────────────────────────────────────
def process_all(
    *,
    events_csv: Path = EVENTS_CSV,
    out_file: Path = OUT_TEX,
    prices_dir: Path = PRICES_DIR,
    csv_file: Path = OUT_CSV,
    allow_three: bool = False,
) -> Path:
    codes, data = collect_outliers(prices_dir)
    events = read_events(events_csv)
    rows = build_rows(codes, data, events)
    if not allow_three:
        for r in rows:
            for c in codes:
                if r[c] == 3:
                    r[c] = 2
    df = pd.DataFrame(rows)[["Row", *codes, "Total"]]
    table = make_table(rows, codes)
    doc = "\n".join(
        [
            "%-------------------------------------------------------------------------------",
            "% fundamentals.tex   v1.5  (2025-06-13)",
            "%-------------------------------------------------------------------------------",
            r"\documentclass[dvipdfmx,oneside]{article}",
            r"\usepackage{amsmath,amssymb,tabularx,booktabs,longtable}",
            r"\usepackage{geometry}",
            r"\geometry{margin=15mm}",
            r"\renewcommand{\arraystretch}{1.2}",
            r"\begin{document}",
            table.rstrip(),
            r"\end{document}",
            "",
        ]
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(doc, encoding="utf-8")
    df.to_csv(csv_file, index=False)
    return out_file

# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="fundamental diff table")
    parser.add_argument("--events", type=Path, default=EVENTS_CSV)
    parser.add_argument("--out", type=Path, default=OUT_TEX)
    parser.add_argument("--prices", type=Path, default=PRICES_DIR)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    parser.add_argument("--allow-three", action="store_true")
    args = parser.parse_args()
    out = process_all(
        events_csv=args.events,
        out_file=args.out,
        prices_dir=args.prices,
        csv_file=args.csv,
        allow_three=args.allow_three,
    )
    print(f"✅ fundamentals → {out.relative_to(out.parent.parent)}")

if __name__ == "__main__":
    main()
