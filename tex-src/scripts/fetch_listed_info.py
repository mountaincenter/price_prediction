#!/usr/bin/env python3
"""scripts/fetch_listed_info.py   v1.0  (2025-06-08)
────────────────────────────────────────────────────────
CHANGELOG:
- 2025-06-08  v1.0 : 初版。listed_info を取得し CSV 保存
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jquantsapi import Client

# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TEX_ROOT = ROOT / "tex-src"
PRICES_DIR = TEX_ROOT / "data" / "prices"
LISTED_DIR = TEX_ROOT / "data" / "listed_info"

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


def fetch_listed(cli: Client, code: str, date: str | None = None) -> pd.DataFrame:
    if date:
        return cli.get_listed_info(code=code, date_yyyymmdd=date)
    return cli.get_listed_info(code=code)


def list_codes() -> list[str]:
    return [p.stem for p in PRICES_DIR.glob("*.csv")]


def main() -> None:
    parser = argparse.ArgumentParser(description="J-Quants listed_info downloader")
    parser.add_argument(
        "--date",
        default=None,
        help="取得基準日 (yyyymmdd)。省略すると最新を取得",
    )
    args = parser.parse_args()

    client = get_client()

    LISTED_DIR.mkdir(parents=True, exist_ok=True)
    for code in list_codes():
        df = fetch_listed(client, code, args.date)
        out = LISTED_DIR / f"{code}.csv"
        df.to_csv(out, index=False)
        logger.info("saved: %s", out)


if __name__ == "__main__":
    main()
