import numpy as np
from sklearn.ensemble import RandomForestRegressor

from features import calendar_features
from holidays import school_holiday_flags
from weather import weather_features

RANDOM_SEED = 42

def build_inputs(previous_day_loads, target_dates, bank_holidays, weather):
    return np.hstack([previous_day_loads,
                      calendar_features(target_dates, bank_holidays),
                      school_holiday_flags(target_dates),
                      weather_features(target_dates, weather)])


def make_model():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))
