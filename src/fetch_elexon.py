# Download UK Actual Total Load from the Elexon API
# API only allows weekly publish window per request so longer ranges are cached as JSON in data

import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/ATL"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_WINDOW = timedelta(days=7)


def _fetch_window(start, end, retries=3):
    params = {
        "publishDateTimeFrom": start.strftime("%Y-%m-%dT%H:%MZ"),
        "publishDateTimeTo": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()["data"]
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# Fetch ATL records from start_date to end_date as a dataframe
def fetch_atl(start_date, end_date, use_cache=True):
    
    start_date, end_date = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
    cache_file = DATA_DIR / f"atl_{start_date}_{end_date}.json"

    if use_cache and cache_file.exists():
        return pd.read_json(cache_file)

    records = []
    window_start = datetime.combine(start_date, datetime.min.time())
    final = datetime.combine(end_date, datetime.min.time())
    while window_start < final:
        # Stops window 1 minute short of 7 day cap
        window_end = min(window_start + MAX_WINDOW - timedelta(minutes=1), final)
        records.extend(_fetch_window(window_start, window_end))
        print(f"  fetched up to {window_end:%Y-%m-%d}  ({len(records)} records)", end="\r")
        window_start = window_start + MAX_WINDOW
    print()

    if not records:
        raise ValueError(f"No data returned for {start_date} to {end_date} ")

    DATA_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(records))
    return pd.DataFrame(records)


if __name__ == "__main__":
    start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("2025-01-01", "2026-01-01")
    df = fetch_atl(start, end, use_cache=False)
    print(f"Saved {len(df)} records for {start} to {end} in {DATA_DIR}")
