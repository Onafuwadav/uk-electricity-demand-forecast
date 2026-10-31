import numpy as np
import pandas as pd


def prepare_dataset(df, verbose=False, return_dates=False):
    # Clean ATL records and build day-ahead pairs
    if verbose:
        print("Duplicate rows: ", df.duplicated(subset=['settlementDate','settlementPeriod']).sum())
    
    # sort with RevisionNumber so when removing dupes it keeps last revised doc
    df = df.sort_values(['settlementDate','settlementPeriod','documentRevisionNumber']) 
    # remove dupes rows with same date and period and keep last
    df = df.drop_duplicates(subset=['settlementDate','settlementPeriod'], keep='last') 

    # Make copy to work on so the dataframe not modified
    df = df.copy()
    df['settlementDate'] = pd.to_datetime(df['settlementDate'])

    if verbose:
        empty_periods = (48 - df.groupby('settlementDate').size()).clip(lower=0).sum()
        print("NaN's + empty periods: ", df.isna().sum().sum() + empty_periods)

    table = df.pivot(index='settlementDate', columns='settlementPeriod', values='quantity')
    table = table.reindex(columns=range(1,49)).sort_index()
    # remove row is date has less then 40 periods
    table = table[table.isna().sum(axis=1) <= 8] 
    table = table.interpolate(axis=1, limit_direction='both')

    consecutive = (table.index.to_numpy()[1:] - table.index.to_numpy()[:-1]) == np.timedelta64(1, 'D') # removal of rows leads to missmatch in dates for training. check which rows need to be discarded

    feature_dataset = table.to_numpy()[:-1]
    label_dataset = table.to_numpy()[1:]

    if return_dates:
        target_dates = table.index[1:][consecutive]
        return feature_dataset[consecutive], label_dataset[consecutive], target_dates
    return feature_dataset[consecutive], label_dataset[consecutive] # remove mismatch days
