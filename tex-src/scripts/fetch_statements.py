#!/usr/bin/env python3
"""scripts/fetch_statements.py   v1.2  (2025-06-08)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-08  v1.2 : fix .env path to repository root
- 2025-06-08  v1.1 : .env から認証情報を読み込み、Refresh Token 対応
- 2025-06-08  v1.0 : 初版。prices 内の銘柄の statements を CSV 保存
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import requests
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TEX_ROOT = ROOT / "tex-src"
PRICES_DIR = TEX_ROOT / "data" / "prices"
EARN_DIR = TEX_ROOT / "data" / "earn"
API_BASE = "https://api.jquants.com/v1"

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_token() -> str:
    refresh = os.getenv("JQUANTS_REFRESH_TOKEN")
    if refresh:
        res = requests.post(
            f"{API_BASE}/token/auth_refresh?refreshtoken={refresh}",
            timeout=10,
        )
        res.raise_for_status()
        return res.json().get("idToken")

    mail = os.getenv("JQUANTS_EMAIL")
    pwd = os.getenv("JQUANTS_PASSWORD")
    if not (mail and pwd):
        raise SystemExit("JQUANTS_EMAIL/PASSWORD not set")

    res = requests.post(
        f"{API_BASE}/token/auth_user",
        data={"mailaddress": mail, "password": pwd},
        timeout=10,
    )
    res.raise_for_status()
    return res.json().get("accessToken")


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

    token = get_token()

    EARN_DIR.mkdir(parents=True, exist_ok=True)
    for code in list_codes():
        df = fetch_statements(token, code, args.date)
        out = EARN_DIR / f"{code}.csv"
        df.to_csv(out, index=False)
        logger.info("saved: %s", out)


if __name__ == "__main__":
    main()
