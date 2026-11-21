# UK Day-Ahead Electricity Demand Forecasting

Predicts tomorrow's national electricity demand, from today's values plus calendar information. The data comes from **Elexon BMRS**.

## Pipelines
1. **Data** ([src/fetch_elexon.py](src/fetch_elexon.py)): downloads the ATL dataset from elexon. The API allows at most 7 days per request, so longer ranges are fetched week by week and cached in `data/`
2. **Cleaning** ([src/data.py](src/data.py)):
   - Remove duplicate settlement periods
   - Remove days missing more than 8/48 periods and interpolate smaller gaps
   - Pair each day with the following day, discarding pairs broken by missing or removed day
3. **Calendar features** ([src/features.py](src/features.py)):
   - Day of week
   - Bank holiday?
   - Christmas period?
   - Time of year given as sin/cos, so season is continuous across New Year
4. **Models** ([src/train.py](src/train.py)):
   - Tomorrow = today baseline vs Ridge regression vs Random Forest vs Extra Trees
   - Trained and tested with and without the calendar features

## Evaluation without look-ahead
All splits respect time order, so a model is never trained on days after the ones it's tested on:
- **Train/test:** the first 80% of 2025 for training, the last 20% for testing.
- **Cross-validation:** `TimeSeriesSplit` is an expanding window so each fold trains on everything before a cut-off and tests on everything straight after
- **Validation:** January to June 2026, which none of the models ever see in training

## Results
Validation on unseen data comparing RMSE values:

| Model | Load values only | + calendar features |
|---|---|---|
| Baseline | 3,951 MW | - |
| Ridge | 3,213 MW | 2,882 MW |
| **Random Forest** | 3,071 MW | **2,854 MW** |

The best model is **28% better than the baseline** with the calendar features helping both models

![Validation forecast](outputs/validation_forecast.png)

## Run it
```bash
pip install -r requirements.txt
cd src
python train.py

```
To fetch a different data range run:
```bash
python src/fetch_elexon.py 2024-01-01 2025-01-01
```

## Improvements
- Add weather inputs since solar output can reduces measured demand
- Train on more history 

## Data sources
- Elexon BMRS Insights API `ATL` dataset used under the [BMRS Data Licence](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/)
- UK bank holidays from [gov.uk](https://www.gov.uk/bank-holidays.json)
