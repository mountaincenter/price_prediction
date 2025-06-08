#!/usr/bin/env python3
"""scripts/xlsx_to_csv_perpbr.py   v1.0  (2025-06-08)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-08  v1.0 : 初版 — perpbr.xlsx 一括 CSV 変換
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
XLSX_DIR = ROOT / "tex-src" / "data" / "earn" / "perpbr"
OUT_CSV = ROOT / "tex-src" / "data" / "earn" / "perpbr.csv"


def read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, header=3)


def main() -> None:
    argparse.ArgumentParser(description="perpbr xlsx to single CSV").parse_args()

    frames: list[pd.DataFrame] = []
    existing_months: set[str] = set()

    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        frames.append(existing)
        if "Year/Month" in existing.columns:
            existing_months.update(existing["Year/Month"].astype(str).unique())

    for xlsx in sorted(XLSX_DIR.glob("perpbr*.xlsx")):
        df = read_xlsx(xlsx)
        month = str(df["Year/Month"].iloc[0])
        if month in existing_months:
            continue
        frames.append(df)

    if not frames:
        print("🟡 更新対象なし")
        return

    merged = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"✅ 出力: {OUT_CSV}")


if __name__ == "__main__":
    main()
