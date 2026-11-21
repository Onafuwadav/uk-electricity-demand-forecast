# Daily weather from Open Meteo forecast API
# Weather history from archive API

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Unweighted average of five major cities
CITIES = {
    "London": (51.51, -0.13),
    "Birmingham": (52.48, -1.90),
    "Manchester": (53.48, -2.24),
    "Leeds": (53.80, -1.55),
    "Glasgow": (55.86, -4.25),
}
DAILY = "temperature_2m_mean,shortwave_radiation_sum"
# if below this temp heating demand grows roughly linearly
HEATING_BASE_C = 15.5


def _request(url, params):
    params = dict(params,
                  latitude=",".join(str(lat) for lat, _ in CITIES.values()),
                  longitude=",".join(str(lon) for _, lon in CITIES.values()),
                  daily=DAILY, timezone="Europe/London")
    
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)

    places = response.json()
    frames = []
    for place in places:
        d = place["daily"]
        frames.append(pd.DataFrame({"temp": d["temperature_2m_mean"], "solar": d["shortwave_radiation_sum"]},
                                   index=pd.to_datetime(d["time"])))
        
    out = pd.concat(frames).groupby(level=0).mean()
    out.index.name = "date"
    return out


def fetch_weather_history(start_date, end_date, use_cache=True):
    start_date, end_date = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
    cache_file = DATA_DIR / f"weather_{start_date}_{end_date}.json"

    if use_cache and cache_file.exists():
        df = pd.read_json(cache_file)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    
    df = _request(ARCHIVE_URL, {"start_date": str(start_date), "end_date": str(end_date)})
    DATA_DIR.mkdir(exist_ok=True)
    cache_file.write_text(df.to_json())

    return df


def fetch_weather_recent(past_days=7, forecast_days=2):
    return _request(FORECAST_URL, {"past_days": past_days, "forecast_days": forecast_days})


def weather_features(target_dates, weather):
    w = weather.reindex(pd.DatetimeIndex(target_dates).normalize())

    if w.isna().any().any():
        missing = w.index[w.isna().any(axis=1)]
        raise ValueError(f"No weather for {len(missing)} day(s), first {missing[0]:%Y-%m-%d}")
    
    temp = w["temp"].to_numpy()
    
    return np.column_stack([temp, np.clip(HEATING_BASE_C - temp, 0, None), w["solar"].to_numpy()])
