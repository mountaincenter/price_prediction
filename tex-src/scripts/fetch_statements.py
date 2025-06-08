#!/usr/bin/env python3
"""scripts/fetch_statements.py   v1.0  (2025-06-08)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-08  v1.0 : 初版。prices 内の銘柄の statements を CSV 保存
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PRICES_DIR = ROOT / "data" / "prices"
EARN_DIR = ROOT / "data" / "earn"
API_BASE = "https://api.jquants.com/v1"


def get_token(mail: str, password: str) -> str:
    res = requests.post(
        f"{API_BASE}/token/auth_user",
        data={"mailaddress": mail, "password": password},
    )
    res.raise_for_status()
    return res.json()["accessToken"]


def fetch_statements(token: str, code: str, date: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}
    params = {"code": code, "date": date}
    res = requests.get(f"{API_BASE}/fins/statements", headers=headers, params=params)
    res.raise_for_status()
    data = res.json()
    if isinstance(data, dict) and "statements" in data:
        records = data["statements"]
    else:
        records = data
    return pd.DataFrame(records)


def list_codes() -> list[str]:
    return [p.stem for p in PRICES_DIR.glob("*.csv")]


def main() -> None:
    parser = argparse.ArgumentParser(description="J-Quants statements downloader")
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y%m%d"),
        help="取得基準日 (yyyymmdd)",
    )
    args = parser.parse_args()

    mail = os.getenv("JQUANTS_EMAIL")
    pwd = os.getenv("JQUANTS_PASSWORD")
    if not (mail and pwd):
        raise SystemExit("JQUANTS_EMAIL/PASSWORD not set")

    token = get_token(mail, pwd)

    EARN_DIR.mkdir(parents=True, exist_ok=True)
    for code in list_codes():
        df = fetch_statements(token, code, args.date)
        out = EARN_DIR / f"{code}.csv"
        df.to_csv(out, index=False)
        print(f"✅ saved: {out}")


if __name__ == "__main__":
    main()
