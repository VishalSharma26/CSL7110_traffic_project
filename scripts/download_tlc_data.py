import argparse
from pathlib import Path

import requests


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TAXI_TYPES = ("yellow", "green", "fhv", "fhvhv")
CHUNK = 1024 * 1024


def make_name(taxi_type, year, month):
    return f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def make_url(name):
    return f"{BASE_URL}/{name}"


def download(url, out_path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0

        with out_path.open("wb") as f:
            for piece in r.iter_content(chunk_size=CHUNK):
                if not piece:
                    continue
                f.write(piece)
                done += len(piece)

                if total:
                    pct = done * 100 / total
                    print(
                        f"Downloaded {done:,} of {total:,} bytes "
                        f"({pct:.1f}%)"
                    )
                else:
                    print(f"Downloaded {done:,} bytes")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download monthly NYC TLC trip data in parquet format."
    )
    parser.add_argument("--taxi-type", choices=TAXI_TYPES, default="yellow")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "raw",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not download the file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the final URL and output path without downloading the file.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 1 <= args.month <= 12:
        raise ValueError("Month must be between 1 and 12.")
    if not 2009 <= args.year <= 2100:
        raise ValueError("Year looks invalid.")


def main():
    args = parse_args()
    validate_args(args)

    name = make_name(args.taxi_type, args.year, args.month)
    url = make_url(name)
    out_dir = args.output_dir.resolve()
    out_path = out_dir / name

    print("Taxi type:", args.taxi_type)
    print("Source URL:", url)
    print("Output path:", out_path)

    if args.dry_run:
        print("Dry run complete. No file downloaded.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and args.skip_existing:
        print("File already exists. Skipping download.")
        return 0

    try:
        download(url, out_path)
    except requests.HTTPError as error:
        print(f"Download failed: {error}")
        if out_path.exists():
            out_path.unlink()
        return 1

    print("Download finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
