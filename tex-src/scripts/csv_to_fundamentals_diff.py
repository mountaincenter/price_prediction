#!/usr/bin/env python3
"""scripts/csv_to_fundamentals_diff.py   v1.1  (2025-06-10)
────────────────────────────────────────────────────────
CHANGELOG:
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

# ──────────────────────────────────────────────────────────────
def read_events(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["Date"] = pd.to_datetime(df["Date"].str.replace("(予定)", "", regex=False))
    df["Code"] = df["国"].map({"日本": "J", "中国": "C", "米国": "U"})
    df["IsTrump"] = df["カテゴリ"].str.contains("トランプ")
    return df[["Date", "Code", "IsTrump"]]

# ──────────────────────────────────────────────────────────────
def collect_outliers(prices_dir: Path) -> tuple[list[str], dict[str, dict[pd.Timestamp, int]]]:
    codes = []
    data: dict[str, dict[pd.Timestamp, int]] = {}
    for csv_path in sorted(prices_dir.glob("*.csv")):
        code = csv_path.stem
        raw = read_prices(csv_path)
        df = calc_center_shift(raw)
        df["Date_full"] = raw["Date"]
        mask = (df["Date_full"] >= START_DATE) & (df["Date_full"] <= END_DATE)
        flags = {
            d.date(): int(o) for d, o in zip(df.loc[mask, "Date_full"], df.loc[mask, "Outlier"])
        }
        codes.append(code)
        data[code] = flags
    return codes, data

# ──────────────────────────────────────────────────────────────
def build_rows(codes: list[str], data: dict[str, dict[pd.Timestamp, int]], events: pd.DataFrame) -> list[dict[str, int]]:
    all_dates = sorted({d for flags in data.values() for d in flags})
    rows = []
    for d in all_dates:
        base = {c: data[c].get(d, 0) for c in codes}
        evs = []
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
            rows.append({"Row": str(d), **base})
        else:
            for t in evs:
                rows.append({"Row": t, **base})
    return rows

# ──────────────────────────────────────────────────────────────
def make_table(rows: list[dict[str, int]], codes: list[str]) -> str:
    df = pd.DataFrame(rows)
    fmt_str = "l" + "c" * len(codes)
    latex = df.to_latex(index=False, escape=False, column_format=fmt_str)
    return "\n".join([r"\begingroup", r"\footnotesize", latex.rstrip(), r"\endgroup"]) + "\n"

# ──────────────────────────────────────────────────────────────
def process_all(
    *,
    events_csv: Path = EVENTS_CSV,
    out_file: Path = OUT_TEX,
    prices_dir: Path = PRICES_DIR,
) -> Path:
    codes, data = collect_outliers(prices_dir)
    events = read_events(events_csv)
    rows = build_rows(codes, data, events)
    table = make_table(rows, codes)
    doc = "\n".join(
        [
            "%\\-------------------------------------------------------------------------------",
            "% fundamentals.tex   v1.1  (2025-06-10)",
            "%-------------------------------------------------------------------------------",
            "\\documentclass[dvipdfmx,oneside]{article}",
            "\\usepackage{amsmath,amssymb,tabularx,booktabs}",
            "\\usepackage{geometry}",
            "\\geometry{margin=15mm}",
            "\\renewcommand{\\arraystretch}{1.2}",
            "\\begin{document}",
            table.rstrip(),
            "\\end{document}",
            "",
        ]
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(doc, encoding="utf-8")
    return out_file

# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="fundamental diff table")
    parser.add_argument("--events", type=Path, default=EVENTS_CSV)
    parser.add_argument("--out", type=Path, default=OUT_TEX)
    parser.add_argument("--prices", type=Path, default=PRICES_DIR)
    args = parser.parse_args()
    out = process_all(events_csv=args.events, out_file=args.out, prices_dir=args.prices)
    print(f"✅ fundamentals → {out.relative_to(out.parent.parent)}")

if __name__ == "__main__":
    main()
