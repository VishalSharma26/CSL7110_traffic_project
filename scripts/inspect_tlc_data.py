import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a TLC parquet file and print a small summary."
    )
    parser.add_argument("--file", type=Path, required=True, help="Path to a parquet file")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of sample rows to display",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.file.resolve()

    if not path.exists():
        print(f"File not found: {path}")
        return 1

    pq_file = pq.ParquetFile(path)
    schema = pq_file.schema_arrow
    meta = pq_file.metadata

    print("File:", path)
    print("Rows:", meta.num_rows)
    print("Row groups:", meta.num_row_groups)
    print("Columns:", len(schema.names))

    print("\nSchema:")
    for name in schema.names:
        print(f"  - {name}: {schema.field(name).type}")

    print("\nSample rows:")
    sample = pq_file.read_row_group(0)
    df = sample.slice(0, args.sample_rows).to_pandas()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
