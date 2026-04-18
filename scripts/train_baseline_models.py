import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
]
TARGET_COLUMN = "trip_duration_minutes"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train simple regression baselines on the cleaned TLC sample."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("reports") / "baseline_metrics.json",
    )
    parser.add_argument(
        "--chart-file",
        type=Path,
        default=Path("reports") / "figures" / "baseline_regression_metrics.png",
    )
    return parser.parse_args()


def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
    }


def save_chart(metrics, chart_file):
    chart_file.parent.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    rmse_vals = [metrics[name]["rmse"] for name in names]
    mae_vals = [metrics[name]["mae"] for name in names]
    x = range(len(names))

    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.15 for i in x], rmse_vals, width=0.3, label="RMSE")
    plt.bar([i + 0.15 for i in x], mae_vals, width=0.3, label="MAE")
    plt.xticks(list(x), names)
    plt.ylabel("Minutes")
    plt.title("Baseline Regression Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150)
    plt.close()


def main():
    args = parse_args()
    in_file = args.input_file.resolve()

    if not in_file.exists():
        print(f"Input file not found: {in_file}")
        return 1

    df = pd.read_parquet(in_file)
    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()

    x = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=60,
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        ),
    }

    metrics = {}
    for name, model in models.items():
        print(f"Training {name}...")
        metrics[name] = evaluate_model(model, x_train, x_test, y_train, y_test)
        print(
            f"{name}: RMSE={metrics[name]['rmse']:.3f}, "
            f"MAE={metrics[name]['mae']:.3f}"
        )

    metrics_path = args.metrics_file.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_chart(metrics, args.chart_file.resolve())
    print("Saved metrics:", metrics_path)
    print("Saved chart:", args.chart_file.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
