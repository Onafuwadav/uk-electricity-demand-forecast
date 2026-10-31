# Calendar features
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BANK_HOLIDAY_URL = "https://www.gov.uk/bank-holidays.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_bank_holidays(division="england-and-wales"):
    # UK bank holiday dates from the gov.uk
    cache_file = DATA_DIR / "bank_holidays.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        response = requests.get(BANK_HOLIDAY_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        DATA_DIR.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps(data))
    return pd.to_datetime([event["date"] for event in data[division]["events"]])


def calendar_features(target_dates, bank_holidays):
    target_dates = pd.DatetimeIndex(target_dates)
    holidays = set(bank_holidays.normalize())

    day_of_week = np.eye(7)[target_dates.day_of_week]
    is_holiday = np.array([d in holidays for d in target_dates], dtype=float)
    was_holiday = np.array([(d - pd.Timedelta(days=1)) in holidays for d in target_dates], dtype=float)
    christmas = (((target_dates.month == 12) & (target_dates.day >= 24)) |
                 ((target_dates.month == 1) & (target_dates.day == 1))).astype(float)
    angle = 2 * np.pi * target_dates.day_of_year.to_numpy() / 365.25

    return np.column_stack([day_of_week, is_holiday, was_holiday, christmas,
                            np.sin(angle), np.cos(angle)])


def add_calendar_features(feature_dataset, target_dates, bank_holidays):
    # Append the calendar features to the load values
    return np.hstack([feature_dataset, calendar_features(target_dates, bank_holidays)])
