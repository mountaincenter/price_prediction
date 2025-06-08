#!/usr/bin/env python3
"""scripts/fetch_statements.py   v1.4  (2025-06-08)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-08  v1.4 : --date 省略時に全期間の statements を取得
- 2025-06-08  v1.3 : jquantsapi.Client で statements を取得
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
from dotenv import load_dotenv
from jquantsapi import Client

# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TEX_ROOT = ROOT / "tex-src"
PRICES_DIR = TEX_ROOT / "data" / "prices"
EARN_DIR = TEX_ROOT / "data" / "earn"

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_client() -> Client:
    refresh = os.getenv("JQUANTS_REFRESH_TOKEN")
    mail = os.getenv("JQUANTS_EMAIL")
    pwd = os.getenv("JQUANTS_PASSWORD")

    if refresh:
        return Client(refresh_token=refresh)
    if mail and pwd:
        return Client(mail_address=mail, password=pwd)

    raise SystemExit("JQUANTS_EMAIL/PASSWORD not set")


def fetch_statements(cli: Client, code: str, date: str | None = None) -> pd.DataFrame:
    if date:
        return cli.get_fins_statements(code=code, date_yyyymmdd=date)
    return cli.get_fins_statements(code=code)


def list_codes() -> list[str]:
    return [p.stem for p in PRICES_DIR.glob("*.csv")]


def main() -> None:
    parser = argparse.ArgumentParser(description="J-Quants statements downloader")
    parser.add_argument(
        "--date",
        default=None,
        help="取得基準日 (yyyymmdd)。省略すると全期間取得",
    )
    args = parser.parse_args()

    client = get_client()

    EARN_DIR.mkdir(parents=True, exist_ok=True)
    for code in list_codes():
        df = fetch_statements(client, code, args.date)
        out = EARN_DIR / f"{code}.csv"
        df.to_csv(out, index=False)
        logger.info("saved: %s", out)


if __name__ == "__main__":
    main()
