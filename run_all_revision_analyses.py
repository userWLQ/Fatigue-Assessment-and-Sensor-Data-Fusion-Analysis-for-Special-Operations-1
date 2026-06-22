#!/usr/bin/env python3
"""Run all revised analysis scripts in sequence.

Run from the repository root:

    python analysis/run_all_revision_analyses.py

The script calls:
    revision_statistical_analysis.py
    cluster_sensitivity_analysis.py
    fatigue_index_analysis.py
    supplementary_tables_generation.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--require-fatigue-data",
        action="store_true",
        help="Fail if fatigue_index.csv is absent.",
    )
    return parser.parse_args()


def run_script(script: Path, root: Path, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, str(script), "--root", str(root)]
    if extra_args:
        command.extend(extra_args)
    print(f"\nRunning: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    script_dir = Path(__file__).resolve().parent

    run_script(script_dir / "revision_statistical_analysis.py", root)
    run_script(script_dir / "cluster_sensitivity_analysis.py", root)

    fatigue_args = ["--require-fatigue-data"] if args.require_fatigue_data else []
    run_script(script_dir / "fatigue_index_analysis.py", root, fatigue_args)
    run_script(script_dir / "supplementary_tables_generation.py", root)

    print("\nAll revised analysis scripts finished.")
    print(f"Statistics: {root / 'outputs' / 'revision_statistics'}")
    print(f"Tables:     {root / 'outputs' / 'revision_tables'}")


if __name__ == "__main__":
    main()
