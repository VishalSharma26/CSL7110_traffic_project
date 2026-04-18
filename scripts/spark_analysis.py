"""
spark_analysis.py
-----------------
Uses Apache Spark to process the full 2.96M-row NYC TLC dataset.
Computes trip statistics, congestion breakdown, and hourly speed averages
that would be too slow to compute repeatedly on a single machine with pandas.

Usage:
    python scripts/spark_analysis.py --input-file data/raw/yellow_tripdata_2024-01.parquet
"""

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spark job: full-dataset summary for NYC TLC yellow taxi data."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/raw/yellow_tripdata_2024-01.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    return parser.parse_args()


def build_session():
    spark = (
        SparkSession.builder
        .appName("TLC_Traffic_Analysis")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def add_features(df):
    """Add derived columns — mirrors build_modeling_sample.py but runs on Spark."""
    df = df.withColumn(
        "trip_duration_minutes",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
    )
    df = df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_weekday", F.dayofweek("tpep_pickup_datetime"))  # 1=Sun, 7=Sat

    df = df.withColumn(
        "trip_speed_mph",
        F.when(
            df["trip_duration_minutes"] > 0,
            df["trip_distance"] / (df["trip_duration_minutes"] / 60.0),
        ).otherwise(None),
    )

    df = df.withColumn(
        "congestion_level",
        F.when(df["trip_speed_mph"] < 8, "high")
         .when((df["trip_speed_mph"] >= 8) & (df["trip_speed_mph"] < 16), "medium")
         .when(df["trip_speed_mph"] >= 16, "low")
         .otherwise("unknown").cast(StringType()),
    )
    return df


def clean_rows(df):
    """Remove invalid/outlier trips."""
    return df.filter(
        (df["trip_duration_minutes"] > 1) &
        (df["trip_duration_minutes"] < 180) &
        (df["trip_distance"] > 0) &
        (df["trip_speed_mph"] > 0) &
        (df["trip_speed_mph"] < 80) &
        (df["fare_amount"] > 0) &
        (df["passenger_count"] > 0)
    )


def main():
    args = parse_args()
    in_file = args.input_file.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        print(f"Input file not found: {in_file}")
        print("Run:  python scripts/download_tlc_data.py --taxi-type yellow --year 2024 --month 1")
        return 1

    print("=" * 60)
    print("Starting Spark session...")
    spark = build_session()
    print(f"Spark version: {spark.version}")

    # ── 1. Load full dataset ────────────────────────────────────────
    print(f"\nReading: {in_file}")
    df_raw = spark.read.parquet(str(in_file))
    total_raw = df_raw.count()
    print(f"Total rows in raw dataset : {total_raw:,}")
    print(f"Columns                   : {len(df_raw.columns)}")

    # ── 2. Feature engineering + cleaning ──────────────────────────
    print("\nEngineering features and cleaning data...")
    df = add_features(df_raw)
    df = clean_rows(df)
    df.cache()                          # keep in memory for multiple passes
    total_clean = df.count()
    print(f"Rows after cleaning       : {total_clean:,}")
    print(f"Rows removed              : {total_raw - total_clean:,} "
          f"({(total_raw - total_clean) / total_raw * 100:.1f}%)")

    # ── 3. Overall trip statistics ──────────────────────────────────
    print("\nComputing overall statistics...")
    stats = df.agg(
        F.round(F.mean("trip_duration_minutes"), 2).alias("avg_duration_min"),
        F.round(F.mean("trip_distance"), 2).alias("avg_distance_miles"),
        F.round(F.mean("trip_speed_mph"), 2).alias("avg_speed_mph"),
        F.round(F.mean("fare_amount"), 2).alias("avg_fare_usd"),
        F.round(F.stddev("trip_duration_minutes"), 2).alias("stddev_duration_min"),
    ).collect()[0]

    overall = {
        "total_raw_rows": total_raw,
        "total_clean_rows": total_clean,
        "rows_removed": total_raw - total_clean,
        "avg_duration_minutes": float(stats["avg_duration_min"]),
        "avg_distance_miles": float(stats["avg_distance_miles"]),
        "avg_speed_mph": float(stats["avg_speed_mph"]),
        "avg_fare_usd": float(stats["avg_fare_usd"]),
        "stddev_duration_minutes": float(stats["stddev_duration_min"]),
    }

    print(f"  Avg trip duration : {overall['avg_duration_minutes']} min")
    print(f"  Avg trip distance : {overall['avg_distance_miles']} miles")
    print(f"  Avg speed         : {overall['avg_speed_mph']} mph")
    print(f"  Avg fare          : ${overall['avg_fare_usd']}")

    # ── 4. Congestion breakdown ─────────────────────────────────────
    print("\nCongestion level breakdown (full dataset)...")
    cong_df = (
        df.groupBy("congestion_level")
          .count()
          .withColumn("percentage", F.round(F.col("count") / total_clean * 100, 1))
          .orderBy("congestion_level")
    )
    cong_df.show()

    congestion = {
        row["congestion_level"]: {
            "count": row["count"],
            "percentage": float(row["percentage"]),
        }
        for row in cong_df.collect()
    }

    # ── 5. Hourly average speed ─────────────────────────────────────
    print("Hourly average speed (all 2.96M trips)...")
    hourly_df = (
        df.groupBy("pickup_hour")
          .agg(
              F.round(F.mean("trip_speed_mph"), 2).alias("avg_speed_mph"),
              F.round(F.mean("trip_duration_minutes"), 2).alias("avg_duration_min"),
              F.count("*").alias("trip_count"),
          )
          .orderBy("pickup_hour")
    )
    hourly_df.show(24, truncate=False)

    hourly = {
        row["pickup_hour"]: {
            "avg_speed_mph": float(row["avg_speed_mph"]),
            "avg_duration_min": float(row["avg_duration_min"]),
            "trip_count": row["trip_count"],
        }
        for row in hourly_df.collect()
    }

    # ── 6. Peak vs off-peak comparison ─────────────────────────────
    print("Peak vs off-peak summary...")
    peak_df = df.withColumn(
        "period",
        F.when(df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]), "rush_hour")
         .otherwise("off_peak")
    ).groupBy("period").agg(
        F.count("*").alias("trips"),
        F.round(F.mean("trip_duration_minutes"), 2).alias("avg_duration_min"),
        F.round(F.mean("trip_speed_mph"), 2).alias("avg_speed_mph"),
    )
    peak_df.show()

    peak = {
        row["period"]: {
            "trips": row["trips"],
            "avg_duration_min": float(row["avg_duration_min"]),
            "avg_speed_mph": float(row["avg_speed_mph"]),
        }
        for row in peak_df.collect()
    }

    # ── 7. Save summary JSON ────────────────────────────────────────
    summary = {
        "source": str(in_file),
        "spark_version": spark.version,
        "overall": overall,
        "congestion_breakdown": congestion,
        "hourly_stats": hourly,
        "peak_vs_offpeak": peak,
    }

    out_path = out_dir / "spark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSpark summary saved to: {out_path}")

    spark.stop()
    print("Spark session stopped.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
