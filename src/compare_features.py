# Test to see how how much weather and holidays help?
# Trained on 2025 data and tested on unseen 2026 data

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from features import calendar_features, fetch_bank_holidays
from fetch_demand import PERIODS, fetch_demand
from holidays import school_holiday_flags
from model import make_model, rmse
from weather import fetch_weather_history, weather_features

END = date.today() - timedelta(days=2)
demand = fetch_demand("2024-12-31", END)
weather = fetch_weather_history("2024-12-31", END)
bank_holidays = fetch_bank_holidays()


def pairs(first, last):
    days = [d for d in demand.index if pd.Timestamp(first) <= d <= pd.Timestamp(last)
            and d - pd.Timedelta(days=1) in demand.index and d in weather.index]
    
    days = pd.DatetimeIndex(days)

    return demand.loc[days - pd.Timedelta(days=1), PERIODS].to_numpy(), demand.loc[days, PERIODS].to_numpy(), days


def inputs(loads, days, school, use_weather):
    parts = [loads, calendar_features(days, bank_holidays)]

    if school:
        parts.append(school_holiday_flags(days))

    if use_weather:
        parts.append(weather_features(days, weather))

    return np.hstack(parts)


X_train, Y_train, train_days = pairs("2025-01-01", "2025-12-31")
X_test, Y_test, test_days = pairs("2026-01-01", END)

print(f"Test: {test_days.min():%d %b %Y} to {test_days.max():%d %b %Y} ({len(test_days)} days), RMSE")
print(f"  {'Baseline (same as yesterday)':<34} {rmse(Y_test, X_test):6.0f} MW")

for name, school, use_weather in [("Calendar", False, False), ("Calendar + school holidays", True, False),
                                  ("Calendar + weather", False, True), ("Calendar + school holidays + weather", True, True)]:
    model = make_model().fit(inputs(X_train, train_days, school, use_weather), Y_train)
    
    print(f"  {name:<34} {rmse(Y_test, model.predict(inputs(X_test, test_days, school, use_weather))):6.0f} MW")
