from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from data import prepare_dataset
from fetch_elexon import fetch_atl
from features import fetch_bank_holidays, add_calendar_features

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Train/test on a full year of public data from Elexon API then validate on later period
TRAIN_PERIOD = ("2025-01-01", "2026-01-01")
VALIDATION_PERIOD = ("2026-01-01", "2026-07-01")

print(f"Loading ATL data for training {TRAIN_PERIOD[0]} to {TRAIN_PERIOD[1]}")
df = fetch_atl(*TRAIN_PERIOD)
print(f"Loading ATL data for validating {VALIDATION_PERIOD[0]} to {VALIDATION_PERIOD[1]}")
df_evaluation = fetch_atl(*VALIDATION_PERIOD)

# Data preparation for adding calendar features
feature_dataset, label_dataset, target_dates = prepare_dataset(df, verbose=True, return_dates=True)

bank_holidays = fetch_bank_holidays()
feature_dataset_cal = add_calendar_features(feature_dataset, target_dates, bank_holidays)

hours = (np.arange(1, 49) / 2) + 0.5 # scaled x-axis to hr incriments instead of 30mins

plt.figure()
plt.plot(hours, feature_dataset[0], label='Feature Set')
plt.plot(hours, label_dataset[0], '--', label='Label Set')
plt.xticks(np.arange(0, 25, step=2))
plt.xlabel('Hr')
plt.ylabel('Load(MW)')
plt.title('Day-ahead forecasting problem')
plt.legend()
plt.grid(alpha=0.3)

# Save figures
plt.savefig(OUTPUT_DIR / "day_ahead_example.png", dpi=120)

print("Feature Shape: ", feature_dataset.shape)
print("Mean: %.1f  Std: %.1f  Min: %.1f  Max: %.1f" % (feature_dataset.mean(), feature_dataset.std(), feature_dataset.min(), feature_dataset.max()))

print("Label Shape: ", label_dataset.shape)
print("Mean: %.1f  Std: %.1f  Min: %.1f  Max: %.1f" % (label_dataset.mean(), label_dataset.std(), label_dataset.min(), label_dataset.max()))

# Ridge regression(80/20 split)
X_train, X_test, Y_train, Y_test = train_test_split(feature_dataset, label_dataset, test_size=0.2, shuffle=False)

cwk_model_part2 = Ridge(alpha=1.5)
cwk_model_part2.fit(X_train, Y_train)

Y_train_output = cwk_model_part2.predict(X_train)
Y_test_output = cwk_model_part2.predict(X_test)

mse_train = mean_squared_error(Y_train, Y_train_output)
mse_test = mean_squared_error(Y_test, Y_test_output)

print("---------------------------------------------------------------------------")
print("Ridge regression:")
print(f"  MSE Train: {mse_train:.4e}")
print(f"  MSE Test : {mse_test:.4e}")
print("---------------------------------------------------------------------------")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(np.arange(1, 49)/2, Y_train[0], label='Actual')
axes[0].plot(np.arange(1, 49)/2, Y_train_output[0], '--', label='Predicted')
axes[0].set_title('Ridge - train set, first row')

axes[1].plot(np.arange(1, 49)/2, Y_test[0], label='Actual')
axes[1].plot(np.arange(1, 49)/2, Y_test_output[0], '--', label='Predicted')
axes[1].set_title('Ridge - test set, first row')

for ax in axes:
    ax.set_xticks(np.arange(2, 25, 2))
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    ax.grid(alpha=0.3)


fig.savefig(OUTPUT_DIR / "ridge_fit.png", dpi=120)

# Random Forest(Same split)
X_train, X_test, Y_train, Y_test = train_test_split(feature_dataset, label_dataset, test_size=0.2, shuffle=False)
cwk_model_part3 = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
cwk_model_part3.fit(X_train, Y_train)

Y_train_output = cwk_model_part3.predict(X_train)
Y_test_output = cwk_model_part3.predict(X_test)

mse_train_p3 = mean_squared_error(Y_train, Y_train_output)
mse_test_p3 = mean_squared_error(Y_test, Y_test_output)

# Baseline set as predict tomorrow = today 
print("Baseline MSE Test: %.4e" % mean_squared_error(Y_test, X_test))
print("---------------------------------------------------------------------------")
print("Random Forest:")
print(f"  MSE Train: {mse_train_p3:.4e}")
print(f"  MSE Test : {mse_test_p3:.4e}")
print("---------------------------------------------------------------------------")

# Same models but now with calendar features
X_train_cal, X_test_cal, Y_train, Y_test = train_test_split(feature_dataset_cal, label_dataset, test_size=0.2, shuffle=False)

ridge_cal = Ridge(alpha=1.5).fit(X_train_cal, Y_train)
forest_cal = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1).fit(X_train_cal, Y_train)

print("---------------------------------------------------------------------------")
print("\nTest MSE with calendar features:")
print(f"  Ridge + calendar        : {mean_squared_error(Y_test, ridge_cal.predict(X_test_cal)):.4e}   (vs load only {mse_test:.4e})")
print(f"  Random Forest + calendar: {mean_squared_error(Y_test, forest_cal.predict(X_test_cal)):.4e}   (vs load only {mse_test_p3:.4e})")
print("---------------------------------------------------------------------------")

## Cross-validated model comparison
cv = TimeSeriesSplit(n_splits=5)

ridge_3 = Ridge(alpha=1.5)
kf_forest_3 = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
kf_trees_3 = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)

print("\nCross-validation (TimeSeriesSplit, 5 folds, with calendar features):")
results = {}
for name, model in [('Ridge', ridge_3),
                    ('RandomForest', kf_forest_3),
                    ("ExtraTrees", kf_trees_3)]:

    scores = -cross_val_score(model, feature_dataset_cal, label_dataset,
                              cv=cv, scoring='neg_mean_squared_error')
    results[name] = scores
    print(f"  {name:<12} - CV mean MSE {scores.mean():.4e},   std {scores.std():.4e}")

diffs = results['RandomForest'] - results['ExtraTrees']
print(f"  Diffs: mean MSE {diffs.mean():.4e},  std {diffs.std():.4e}")
print(f"  Forest is better in {(diffs < 0).sum()} out of {len(diffs)} folds")

# Validation on unseen data
feature_dataset_validation, label_dataset_validation, validation_dates = prepare_dataset(df_evaluation, return_dates=True)
feature_dataset_validation_cal = add_calendar_features(feature_dataset_validation, validation_dates, bank_holidays)


print(f"\nValidation on {VALIDATION_PERIOD[0]} to {VALIDATION_PERIOD[1]} ({len(feature_dataset_validation)} days):")
validation_results = {
    "Baseline (tomorrow = today)": feature_dataset_validation,
    "Ridge": cwk_model_part2.predict(feature_dataset_validation),
    "Random Forest": cwk_model_part3.predict(feature_dataset_validation),
    "Ridge + calendar": ridge_cal.predict(feature_dataset_validation_cal),
    "Random Forest + calendar": forest_cal.predict(feature_dataset_validation_cal),
}
for name, prediction in validation_results.items():
    mse_validation = mean_squared_error(label_dataset_validation, prediction)
    print(f"  {name:<28} MSE {mse_validation:.4e}   RMSE {np.sqrt(mse_validation):7.0f} MW")

day = int(np.argmax(validation_dates.day_of_week == 0))
plt.figure(figsize=(8, 4))
plt.plot(hours, label_dataset_validation[day], 'k', label='Actual')
for name in ["Baseline (tomorrow = today)", "Random Forest", "Random Forest + calendar"]:
    plt.plot(hours, validation_results[name][day], '--', label=name)
plt.xticks(np.arange(0, 25, step=2))
plt.xlabel('Hour of day')
plt.ylabel('Load (MW)')
plt.title(f'Day-ahead forecast for Monday {validation_dates[day]:%d %b %Y} (unseen data)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_forecast.png", dpi=120)
print(f"\nPlots saved in {OUTPUT_DIR}")
