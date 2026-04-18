"""
build_modeling_sample.py
------------------------
Uses Apache Spark to read the full raw TLC Parquet file, engineer features,
clean invalid rows, and save a processed sample for model training.

Running on Spark means this script can handle the full 2.96M-row dataset
(or years of data) without running out of memory — unlike a pandas approach.

Usage:
    python scripts/build_modeling_sample.py \
        --input-file  data/raw/yellow_tripdata_2024-01.parquet \
        --output-file data/processed/sample_200k.parquet \
        --max-rows    200000
"""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a cleaned modeling sample from a TLC parquet file using Spark."
    )
    parser.add_argument("--input-file",  type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Number of rows to keep after cleaning (0 = keep all).",
    )
    return parser.parse_args()


def build_session():
    spark = (
        SparkSession.builder
        .appName("TLC_Build_Modeling_Sample")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def add_features(df):
    """Engineer all columns needed for model training."""

    # Trip duration in minutes
    df = df.withColumn(
        "trip_duration_minutes",
        (F.unix_timestamp("tpep_dropoff_datetime") -
         F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
    )

    # Time-based features
    df = df.withColumn("pickup_hour",    F.hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_month",   F.month("tpep_pickup_datetime"))

    # Spark dayofweek: 1=Sunday … 7=Saturday
    # Convert to match pandas convention: 0=Monday … 6=Sunday
    df = df.withColumn(
        "pickup_weekday",
        ((F.dayofweek("tpep_pickup_datetime") + 5) % 7),
    )

    # Weekend flag  (5=Saturday, 6=Sunday in 0-based weekday)
    df = df.withColumn(
        "is_weekend",
        F.when(df["pickup_weekday"].isin([5, 6]), 1).otherwise(0),
    )

    # Rush-hour flag  (7–9 am and 4–7 pm)
    df = df.withColumn(
        "is_rush_hour",
        F.when(df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]), 1).otherwise(0),
    )

    # Speed in mph
    df = df.withColumn(
        "trip_speed_mph",
        F.when(
            df["trip_duration_minutes"] > 0,
            df["trip_distance"] / (df["trip_duration_minutes"] / 60.0),
        ).otherwise(None),
    )

    # Congestion level based on speed
    df = df.withColumn(
        "congestion_level",
        F.when(df["trip_speed_mph"] < 8,  "high")
         .when((df["trip_speed_mph"] >= 8) & (df["trip_speed_mph"] < 16), "medium")
         .when(df["trip_speed_mph"] >= 16, "low")
         .otherwise("unknown")
         .cast(StringType()),
    )

    return df


def clean_rows(df):
    """Remove invalid and extreme-outlier trips."""
    return df.filter(
        (F.col("trip_duration_minutes") > 1)   &
        (F.col("trip_duration_minutes") < 180)  &
        (F.col("trip_distance")         > 0)    &
        (F.col("trip_speed_mph")        > 0)    &
        (F.col("trip_speed_mph")        < 80)   &
        (F.col("fare_amount")           > 0)    &
        (F.col("passenger_count")       > 0)
    )


def main():
    args  = parse_args()
    in_file  = args.input_file.resolve()
    out_file = args.output_file.resolve()

    if not in_file.exists():
        print(f"Input file not found: {in_file}")
        print("Run: python scripts/download_tlc_data.py --taxi-type yellow --year 2024 --month 1")
        return 1

    spark = build_session()
    print(f"Spark version : {spark.version}")
    print(f"Reading       : {in_file}")

    # ── 1. Load ────────────────────────────────────────────────────
    df = spark.read.parquet(str(in_file))
    total_raw = df.count()
    print(f"Raw rows      : {total_raw:,}")

    # ── 2. Feature engineering ─────────────────────────────────────
    df = add_features(df)

    # ── 3. Cleaning ────────────────────────────────────────────────
    df = clean_rows(df)
    total_clean = df.count()
    print(f"After cleaning: {total_clean:,}  "
          f"({total_raw - total_clean:,} removed)")

    # ── 4. Sample ──────────────────────────────────────────────────
    if args.max_rows and args.max_rows < total_clean:
        # Use a reproducible fraction so sampling is deterministic
        fraction = args.max_rows / total_clean
        df = df.sample(fraction=fraction, seed=42)
        sampled = df.count()
        print(f"After sampling: {sampled:,}  (target {args.max_rows:,})")

    # ── 5. Save ────────────────────────────────────────────────────
    # coalesce(1) writes a single Parquet file instead of many part-files,
    # so downstream pandas scripts can read it with pd.read_parquet() as before.
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file/folder at output path so Spark doesn't error
    if out_file.exists():
        import shutil
        if out_file.is_dir():
            shutil.rmtree(out_file)
        else:
            out_file.unlink()

    df.coalesce(1).write.parquet(str(out_file))

    # Rename the part file to the exact output path expected downstream
    import glob as _glob
    part_files = _glob.glob(str(out_file / "part-*.parquet"))
    if part_files:
        import shutil, os
        tmp = out_file.parent / "_tmp_sample.parquet"
        shutil.move(part_files[0], tmp)
        shutil.rmtree(out_file)
        os.rename(tmp, out_file)

    print(f"Saved to      : {out_file}")

    # ── 6. Congestion distribution ─────────────────────────────────
    print("\nCongestion label distribution:")
    cong = df.groupBy("congestion_level").count()
    total = df.count()
    cong = cong.withColumn("pct", F.round(F.col("count") / total, 4))
    cong.orderBy("congestion_level").show()

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
