import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a cleaned modeling sample from a TLC parquet file."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Maximum number of rows to load for the first modeling sample.",
    )
    return parser.parse_args()


def add_features(df):
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])

    df = df.copy()
    df["trip_duration_minutes"] = (dropoff - pickup).dt.total_seconds() / 60.0
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_weekday"] = pickup.dt.dayofweek
    df["pickup_month"] = pickup.dt.month
    df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["trip_speed_mph"] = df["trip_distance"] / (df["trip_duration_minutes"] / 60.0)

    speed = df["trip_speed_mph"]
    rules = [
        speed < 8,
        (speed >= 8) & (speed < 16),
        speed >= 16,
    ]
    labels = ["high", "medium", "low"]
    df["congestion_level"] = np.select(rules, labels, default="unknown")
    return df


def clean_rows(df):
    df = df.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
        ]
    ).copy()

    df = df[
        (df["trip_duration_minutes"] > 1)
        & (df["trip_duration_minutes"] < 180)
        & (df["trip_distance"] > 0)
        & (df["trip_speed_mph"] > 0)
        & (df["trip_speed_mph"] < 80)
        & (df["fare_amount"] > 0)
        & (df["passenger_count"] > 0)
    ]
    return df


def main():
    args = parse_args()
    in_file = args.input_file.resolve()
    out_file = args.output_file.resolve()

    if not in_file.exists():
        print(f"Input file not found: {in_file}")
        return 1

    print("Reading file:", in_file)
    df = pd.read_parquet(in_file)

    if args.max_rows:
        df = df.head(args.max_rows).copy()

    print("Rows loaded:", len(df))
    df = add_features(df)
    clean_df = clean_rows(df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_file, index=False)

    print("Rows after cleaning:", len(clean_df))
    print("Saved file:", out_file)
    print("\nCongestion label distribution:")
    print(clean_df["congestion_level"].value_counts(normalize=True).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
