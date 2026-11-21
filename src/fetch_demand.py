# National demand outturn (published within the hour) from the Elexon BMRS API half-hourly in MW

import time
from datetime import date, timedelta

import pandas as pd
import requests

API_URL = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
MAX_DAYS = 28
PERIODS = [f"p{i}" for i in range(1, 49)]


def _get(start, end, retries=3):
    params = {"settlementDateFrom": str(start), "settlementDateTo": str(end), "format": "json"}

    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            return response.json()["data"]
        
        except requests.RequestException:

            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_demand(start_date, end_date):
    
    start_date, end_date = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
    records, chunk = [], start_date

    while chunk <= end_date:
        chunk_end = min(chunk + timedelta(days=MAX_DAYS - 1), end_date)
        records.extend(_get(chunk, chunk_end))
        chunk = chunk_end + timedelta(days=1)

    if not records:
        return pd.DataFrame(columns=PERIODS)
    
    df = pd.DataFrame(records).drop_duplicates(subset=["settlementDate", "settlementPeriod"], keep="last")
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    table = df.pivot(index="settlementDate", columns="settlementPeriod", values="initialDemandOutturn")
    table = table.reindex(columns=range(1, 49)).sort_index()
    
    table = table[table.isna().sum(axis=1) <= 8].interpolate(axis=1, limit_direction="both")
    table.columns = PERIODS
    table.index.name = "date"
    return table
