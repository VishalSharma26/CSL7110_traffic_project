import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "pyspark",
    "pyarrow",
    "requests",
]
FOLDERS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "interim",
    ROOT / "data" / "processed",
    ROOT / "docs",
    ROOT / "reports",
    ROOT / "scripts",
]


def check_python():
    print("Python version:", sys.version.replace("\n", " "))


def check_java():
    java_home = os.environ.get("JAVA_HOME", "")
    print("JAVA_HOME:", java_home or "not set")

    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("Java command: not found")
        return

    output = result.stderr.strip() or result.stdout.strip() or "no version output"
    line = output.splitlines()[0]
    print("Java version:", line)


def check_modules():
    missing = []
    print("\nPackage check:")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:
            missing.append(name)
            print(f"  [MISSING] {name}")
            continue

        version = getattr(module, "__version__", "unknown")
        print(f"  [OK] {name} {version}")
    return missing


def check_folders():
    print("\nDirectory check:")
    for folder in FOLDERS:
        status = "OK" if folder.exists() else "MISSING"
        print(f"  [{status}] {folder}")


def main():
    print("Traffic Project Environment Check")
    print("=" * 34)
    check_python()
    check_java()
    missing = check_modules()
    check_folders()

    if missing:
        print("\nEnvironment status: incomplete")
        print("Install missing packages before moving to the modeling step.")
        return 1

    print("\nEnvironment status: ready for the next step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
