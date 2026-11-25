import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from model import build_inputs, make_model, rmse

from features import fetch_bank_holidays
from fetch_demand import fetch_demand
from weather import fetch_weather_history, fetch_weather_recent

UK = ZoneInfo("Europe/London")
ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "live"
SITE = ROOT / "site"
PERIODS = [f"p{i}" for i in range(1, 49)]

TRAIN_DAYS = 365
HOLDOUT_DAYS = 28
MIN_HOLDOUT_DAYS = 14
RUN_HOUR = 3


def _read_csv(name, index_col="date"):
    path = LIVE / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=index_col, parse_dates=[index_col])


def _write_csv(df, name):
    LIVE.mkdir(exist_ok=True)
    df.sort_index().to_csv(LIVE / name, float_format="%.1f")


def _read_models():
    path = LIVE / "models.json"
    return json.loads(path.read_text()) if path.exists() else []


def _write_models(models):
    LIVE.mkdir(exist_ok=True)
    (LIVE / "models.json").write_text(json.dumps(models, indent=2) + "\n")


def current_model(models):
    accepted = [m for m in models if m["accepted"]]
    return accepted[-1] if accepted else None


def update_history(today):
    history = _read_csv("history.csv")
    yesterday = today - timedelta(days=1)

    first = (history.index.max().date() + timedelta(days=1)) if len(history) else yesterday - timedelta(days=TRAIN_DAYS + 40)
    if first <= yesterday:

        print(f"Fetching demand for {first} to {yesterday}")
        new = fetch_demand(first, yesterday)

        history = pd.concat([history, new])
        history = history[~history.index.duplicated(keep="last")]
        _write_csv(history, "history.csv")

        print(f"  {len(new)} new day(s), history now {len(history)} days")

    weather = _read_csv("weather.csv")

    if len(weather) == 0:
        start = history.index.min().date()
        weather = fetch_weather_history(start, yesterday - timedelta(days=7), use_cache=False)

    recent = fetch_weather_recent(past_days=10, forecast_days=2)
    weather = pd.concat([weather, recent])
    weather = weather[~weather.index.duplicated(keep="last")]
    observed = weather[weather.index.date < today]
    _write_csv(observed, "weather.csv")

    return history, observed, recent


def training_pairs(history, weather, bank_holidays, first_target, last_target):
    days = history.index
    pairs = [d for d in days if first_target <= d <= last_target and (d - pd.Timedelta(days=1)) in days]

    targets = pd.DatetimeIndex(pairs)
    previous = history.loc[targets - pd.Timedelta(days=1), PERIODS].to_numpy()

    X = build_inputs(previous, targets, bank_holidays, weather)
    Y = history.loc[targets, PERIODS].to_numpy()

    return X, Y, targets


def fit(version, history, weather, bank_holidays):
    X, Y, _ = training_pairs(history, weather, bank_holidays,
                             pd.Timestamp(version["train_start"]), pd.Timestamp(version["train_end"]))
    
    return make_model().fit(X, Y)


def retrain(today, history, weather, bank_holidays, models):
    yesterday = pd.Timestamp(today - timedelta(days=1))
    live = current_model(models)

    attempt = {"version": len([m for m in models if m["accepted"]]) + 1,
               "created": str(today),
               "train_start": str((yesterday - pd.Timedelta(days=TRAIN_DAYS - 1)).date()),
               "train_end": str(yesterday.date())}

    if live is None:
        attempt.update(accepted=True, note="first model")
        models.append(attempt)

        print(f"Model v{attempt['version']}: first model, trained {attempt['train_start']} to {attempt['train_end']}")
        return

    holdout_start = max(yesterday - pd.Timedelta(days=HOLDOUT_DAYS - 1), pd.Timestamp(live["train_end"]) + pd.Timedelta(days=1))
    X_hold, Y_hold, hold_days = training_pairs(history, weather, bank_holidays, holdout_start, yesterday)

    if len(hold_days) < MIN_HOLDOUT_DAYS:
        attempt.update(version=None, accepted=False, note=f"skipped, only {len(hold_days)} unseen days to compare on")
        models.append(attempt)

        print(f"Retrain skipped: only {len(hold_days)} unseen days since the live model was trained")
        return

    candidate = make_model()
    X_c, Y_c, _ = training_pairs(history, weather, bank_holidays,
                                 holdout_start - pd.Timedelta(days=TRAIN_DAYS), holdout_start - pd.Timedelta(days=1))
    candidate.fit(X_c, Y_c)
    live_model = fit(live, history, weather, bank_holidays)
    rmse_candidate = rmse(Y_hold, candidate.predict(X_hold))
    rmse_live = rmse(Y_hold, live_model.predict(X_hold))
    accepted = rmse_candidate < rmse_live
    attempt.update(accepted=accepted,
                   holdout=f"{hold_days.min():%Y-%m-%d} to {hold_days.max():%Y-%m-%d} ({len(hold_days)} days)",
                   holdout_rmse_new=round(rmse_candidate), holdout_rmse_live=round(rmse_live),
                   replaces=live["version"])
    if not accepted:
        attempt["version"] = None
    models.append(attempt)
    verdict = f"accepted as v{attempt['version']}" if accepted else f"rejected, v{live['version']} stays live"
    print(f"Retrain: new {rmse_candidate:.0f} MW vs live {rmse_live:.0f} MW on unseen days, {verdict}")



def score(history, forecasts):
    scores = _read_csv("scores.csv")
    for day, row in forecasts.iterrows():
        if day in history.index and day not in scores.index:
            actual = history.loc[day, PERIODS].to_numpy(dtype=float)
            predicted = row[PERIODS].to_numpy(dtype=float)
            baseline = row[[f"b{p}" for p in PERIODS]].to_numpy(dtype=float)
            new = pd.DataFrame([{"model_version": int(row["model_version"]), "rmse_model": rmse(actual, predicted),
                                 "rmse_baseline": rmse(actual, baseline), "peak_actual": actual.max(),
                                 "peak_predicted": predicted.max()}], index=pd.DatetimeIndex([day], name="date"))
            scores = pd.concat([scores, new]) if len(scores) else new
            print(f"Scored {day:%Y-%m-%d}: model {new['rmse_model'].iloc[0]:.0f} MW, baseline {new['rmse_baseline'].iloc[0]:.0f} MW")
    if len(scores):
        scores.index.name = "date"
        _write_csv(scores, "scores.csv")


def daily(force=False, force_retrain=False):
    now = datetime.now(UK)
    today = now.date()
    
    if not force and now.hour < RUN_HOUR:
        print(f"Before {RUN_HOUR}am in the UK (it is {now:%H:%M})")
        return False
    forecasts = _read_csv("forecasts.csv")
    if pd.Timestamp(today) in forecasts.index and not force:
        print(f"Already forecast {today}")
        return False

    history, weather, recent = update_history(today)
    bank_holidays = fetch_bank_holidays()
    score(history, forecasts)

    models = _read_models()
    if current_model(models) is None or today.day == 1 or force_retrain:
        retrain(today, history, weather, bank_holidays, models)
        _write_models(models)
    live = current_model(models)

    yesterday = pd.Timestamp(today - timedelta(days=1))
    if yesterday not in history.index:
        print(f"No complete demand data for {yesterday:%Y-%m-%d} yet, so no forecast today")
        return True
    model = fit(live, history, weather, bank_holidays)
    target = pd.DatetimeIndex([pd.Timestamp(today)])
    all_weather = pd.concat([weather, recent[recent.index.date >= today]])
    X = build_inputs(history.loc[[yesterday], PERIODS].to_numpy(), target, bank_holidays, all_weather)
    prediction = model.predict(X)[0]

    row = {"model_version": live["version"], "made_at": now.strftime("%Y-%m-%d %H:%M"),
           "temp_forecast": all_weather.loc[target[0], "temp"], "solar_forecast": all_weather.loc[target[0], "solar"]}
    row.update(dict(zip(PERIODS, prediction)))
    row.update({f"b{p}": v for p, v in zip(PERIODS, history.loc[yesterday, PERIODS])})
    new = pd.DataFrame([row], index=pd.DatetimeIndex(target, name="date"))
    forecasts = pd.concat([forecasts[forecasts.index != target[0]], new]) if len(forecasts) else new
    _write_csv(forecasts, "forecasts.csv")
    print(f"Forecast for {today} made with model v{live['version']}: peak {prediction.max():.0f} MW")
    return True


def build_site():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SITE.mkdir(exist_ok=True)
    scores, forecasts, history, models = _read_csv("scores.csv"), _read_csv("forecasts.csv"), _read_csv("history.csv"), _read_models()
    hours = np.arange(1, 49) / 2

    fig, ax = plt.subplots(figsize=(10, 4))
    if len(scores):
        ax.plot(scores.index, scores["rmse_baseline"], "o-", color="#999", ms=3, lw=1, label="Baseline (same as yesterday)")
        ax.plot(scores.index, scores["rmse_model"], "o-", color="#8523DB", ms=3, lw=1.5, label="Model")
        for m in models:
            if m["accepted"] and pd.Timestamp(m["created"]) >= scores.index.min():
                ax.axvline(pd.Timestamp(m["created"]), color="#8523DB", ls="--", lw=1, alpha=0.6)
                ax.text(pd.Timestamp(m["created"]), ax.get_ylim()[1], f" v{m['version']}", va="top", fontsize=9, color="#8523DB")
    ax.set_ylabel("Daily error, RMSE (MW)")
    ax.set_title("How far off each day's forecast was")
    if len(scores):
        ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SITE / "daily_error.png", dpi=120)
    plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    if len(scores):
        day = scores.index.max()
        f = forecasts.loc[day]
        axes[0].plot(hours, history.loc[day, PERIODS], "k", label="Actual")
        axes[0].plot(hours, f[PERIODS].astype(float), "--", color="#8523DB", label="Forecast")
        axes[0].plot(hours, f[[f"b{p}" for p in PERIODS]].astype(float), ":", color="#999", label="Baseline")
        axes[0].set_title(f"{day:%a %d %b %Y}: error {scores.loc[day, 'rmse_model']:.0f} MW")
        axes[0].legend()
    if len(forecasts):
        day = forecasts.index.max()
        axes[1].plot(hours, forecasts.loc[day, PERIODS].astype(float), "--", color="#8523DB", label="Forecast")
        axes[1].set_title(f"Forecast for {day:%a %d %b %Y}")
        axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("Hour (UK time)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Demand (MW)")
    fig.tight_layout()
    fig.savefig(SITE / "latest.png", dpi=120)
    plt.close(fig)


    live = current_model(models)
    summary = "No days scored yet. The first score appears the morning after the first forecast."
    rows = ""
    if len(scores):
        better = 100 * (1 - scores["rmse_model"].mean() / scores["rmse_baseline"].mean())
        summary = (f"Over {len(scores)} day(s) the model's average error is <b>{scores['rmse_model'].mean():,.0f} MW</b>, "
                   f"against {scores['rmse_baseline'].mean():,.0f} MW for the baseline, {better:.0f}% lower.")
        for day, s in scores.sort_index(ascending=False).head(14).iterrows():
            rows += (f"<tr><td>{day:%a %d %b}</td><td>v{int(s['model_version'])}</td><td>{s['rmse_model']:,.0f}</td>"
                     f"<td>{s['rmse_baseline']:,.0f}</td><td>{s['peak_predicted']:,.0f} / {s['peak_actual']:,.0f}</td></tr>")
    log = ""
    for m in reversed(models):
        result = m.get("note") or f"{m['holdout_rmse_new']:,} vs {m['holdout_rmse_live']:,} MW on {m['holdout']}"
        outcome = f"v{m['version']}" if m["accepted"] else ("skipped" if "skipped" in m.get("note", "") else "kept old model")
        log += (f"<tr><td>{m['created']}</td><td>{outcome}</td>"
                f"<td>{m['train_start']} to {m['train_end']}</td><td>{result}</td></tr>")
    updated = datetime.now(UK).strftime("%d %b %Y %H:%M")
    (SITE / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UK Demand Forecast</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 0 auto; padding: 16px; color: #1d1a20; background: #fff; line-height: 1.5; }}
img {{ width: 100%; height: auto; }} table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin-bottom: 24px; }}
th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #ddd; }} .dim {{ color: #666; font-size: 14px; }}
.scroll {{ overflow-x: auto; }}
</style></head><body>
<h1>UK day-ahead electricity demand forecast</h1>
<p>Every night at 3am a GitHub Actions job forecasts the day's national demand, half-hour by half-hour, from yesterday's
demand, the calendar, school holidays and the weather forecast. The next night it scores that forecast against what
actually happened. The baseline simply assumes today will look like yesterday.</p>
<p>{summary}</p>
<p class="dim">Live model: v{live['version'] if live else '-'}, trained on {live['train_start'] if live else '-'} to {live['train_end'] if live else '-'}.
Updated {updated} UK time. Data: Elexon BMRS (national demand outturn), Open-Meteo (weather).</p>
<img src="daily_error.png" alt="Daily forecast error of the model and the baseline">
<img src="latest.png" alt="Latest scored day and today's forecast">
<h2>Last 14 days (MW)</h2>
<div class="scroll"><table><tr><th>Day</th><th>Model</th><th>Model error</th><th>Baseline error</th><th>Peak forecast / actual</th></tr>{rows}</table></div>
<h2>Model versions</h2>
<p class="dim">On the 1st of each month a new model is trained on the latest year. It only replaces the live model if it
was more accurate over the last few weeks, which neither model trained on.</p>
<div class="scroll"><table><tr><th>Date</th><th>Result</th><th>Trained on</th><th>Check (new vs live RMSE)</th></tr>{log}</table></div>
<p class="dim"><a href="https://github.com/Onafuwadav/uk-electricity-demand-forecast">Source on GitHub</a></p>
</body></html>
""")
    print(f"Site written to {SITE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["daily", "site"])
    parser.add_argument("--force", action="store_true", help="Run now")
    parser.add_argument("--retrain", action="store_true", help="Retrain now")
    args = parser.parse_args()
    if args.command == "daily":
        ran = daily(force=args.force, force_retrain=args.retrain)
        github_output = __import__("os").environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as fh:
                fh.write(f"ran={'true' if ran else 'false'}\n")
    else:
        build_site()
