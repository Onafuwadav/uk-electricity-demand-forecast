# Rough school holiday periods

from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter


def _monday_on_or_after(d):
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _last_monday(year, month):
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=d.weekday())


def holiday_ranges(year):
    
    good_friday = easter(year) - timedelta(days=2)
    easter_start = good_friday - timedelta(days=4)
    
    feb_half_term = _monday_on_or_after(date(year, 2, 12))
    may_half_term = _last_monday(year, 5) 
    oct_half_term = _last_monday(year, 10)

    return [
        (date(year - 1, 12, 21), date(year, 1, 3)), 
        (feb_half_term, feb_half_term + timedelta(days=4)),
        (easter_start, easter_start + timedelta(days=11)),
        (may_half_term, may_half_term + timedelta(days=4)),
        (date(year, 7, 22), date(year, 8, 31)),
        (oct_half_term, oct_half_term + timedelta(days=4)),
        (date(year, 12, 21), date(year + 1, 1, 3)),
    ]


def school_holiday_flags(target_dates):
    target_dates = pd.DatetimeIndex(target_dates)
    ranges = {}
    flags = []

    for d in target_dates.date:
        if d.year not in ranges:
            ranges[d.year] = holiday_ranges(d.year)
        flags.append(any(start <= d <= end for start, end in ranges[d.year]))
        
    return np.array(flags, dtype=float).reshape(-1, 1)
