#!/usr/bin/env python3
"""scripts/csv_to_outlier_diff.py
  v1.2  (2025-06-13)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-13  v1.2 : read_statement_events 実装、Outlier=0-8 対応
- 2025-06-13  v1.1 : 月末月初/SQ判定 (5,6) 対応
- 2025-06-13  v1.0 : fundamentals/markets/statements の外れ値を集約
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from csv_to_center_shift_diff import (
    read_prices,
    calc_center_shift,
    read_statement_events,
    read_macro_events,
)

START_DATE = pd.Timestamp("2024-01-04")
END_DATE   = pd.Timestamp("2025-06-09")

PRICES_DIR    = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data" / "prices"
EVENTS_CSV    = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data" / "events.csv"
STATEMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "tex-src" / "data" / "earn"
OUT_TEX       = Path(__file__).resolve().parent.parent.parent / "tex-src" / "outliers.tex"
OUT_CSV       = OUT_TEX.with_suffix('.csv')


def read_events(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["Date"] = pd.to_datetime(df["Date"].str.replace("(予定)", "", regex=False))
    df["Code"] = df["国"].map({"日本": "J", "中国": "C", "米国": "U"})
    df["IsTrump"] = df["カテゴリ"].str.contains("トランプ")
    return df[["Date", "Code", "IsTrump"]]


def collect_outliers(prices_dir: Path, statements_dir: Path) -> tuple[list[str], dict[str, dict[pd.Timestamp, int]]]:
    codes: list[str] = []
    data: dict[str, dict[pd.Timestamp, int]] = {}
    for csv_path in sorted(prices_dir.glob("*.csv")):
        code = csv_path.stem
        raw = read_prices(csv_path)
        stmt_dates = read_statement_events(code, statements_dir)
        df = calc_center_shift(raw, events_csv=EVENTS_CSV, statement_dates=stmt_dates)
        df["Date_full"] = raw["Date"]
        mask = (df["Date_full"] >= START_DATE) & (df["Date_full"] <= END_DATE)
        flags = {
            d.date(): int(o) for d, o in zip(df.loc[mask, "Date_full"], df.loc[mask, "Outlier"])
        }
        codes.append(code)
        data[code] = flags
    return codes, data


def build_rows(codes: list[str], data: dict[str, dict[pd.Timestamp, int]], events: pd.DataFrame) -> list[dict[str, int]]:
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


def make_table(rows: list[dict[str, int]], codes: list[str]) -> str:
    df = pd.DataFrame(rows)
    df = df[["Row", *codes, "Total"]]
    df["Row"] = df["Row"].str.replace("_", r"\_", regex=False)
    fmt_str = "l" + "c" * (len(codes) + 1)
    latex = df.to_latex(index=False, escape=False, column_format=fmt_str, longtable=True)
    return "\n".join([
        r"\begingroup",
        r"\footnotesize",
        latex.rstrip(),
        r"\endgroup",
        "",
    ])


def process_all(
    *,
    events_csv: Path = EVENTS_CSV,
    statements_dir: Path = STATEMENTS_DIR,
    out_file: Path = OUT_TEX,
    prices_dir: Path = PRICES_DIR,
    csv_file: Path = OUT_CSV,
) -> Path:
    codes, data = collect_outliers(prices_dir, statements_dir)
    events = read_events(events_csv)
    rows = build_rows(codes, data, events)
    df = pd.DataFrame(rows)[["Row", *codes, "Total"]]
    table = make_table(rows, codes)
    doc = "\n".join([
        "%--------------------------------------------------------------------------",
        "% outliers.tex   v1.0  (2025-06-13)",
        "%--------------------------------------------------------------------------",
        r"\documentclass[dvipdfmx,oneside]{article}",
        r"\usepackage{amsmath,amssymb,tabularx,booktabs,longtable}",
        r"\usepackage{geometry}",
        r"\geometry{margin=15mm}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{document}",
        table.rstrip(),
        r"\end{document}",
        "",
    ])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(doc, encoding="utf-8")
    df.to_csv(csv_file, index=False)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="outlier diff table")
    parser.add_argument("--events", type=Path, default=EVENTS_CSV)
    parser.add_argument("--statements", type=Path, default=STATEMENTS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_TEX)
    parser.add_argument("--prices", type=Path, default=PRICES_DIR)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    out = process_all(
        events_csv=args.events,
        statements_dir=args.statements,
        out_file=args.out,
        prices_dir=args.prices,
        csv_file=args.csv,
    )
    print(f"✅ outliers → {out.relative_to(out.parent.parent)}")


if __name__ == "__main__":
    main()

