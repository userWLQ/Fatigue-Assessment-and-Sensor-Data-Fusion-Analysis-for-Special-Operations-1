#!/usr/bin/env python3
"""Generate supplementary statistical tables from revised analysis outputs.

This script collects the CSV files produced by the revised analysis scripts and
combines them into one Excel workbook. If Excel writing is unavailable, it also
keeps all source CSV files in place.

Run after:
    python analysis/revision_statistical_analysis.py
    python analysis/cluster_sensitivity_analysis.py
    python analysis/fatigue_index_analysis.py

or simply run:
    python analysis/run_all_revision_analyses.py

Outputs are written to:
    outputs/revision_tables/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TABLE_SPECS = [
    ("S1_descriptive_statistics", "descriptive_statistics.csv"),
    ("S2_normality_tests", "normality_tests.csv"),
    ("S3_groupwise_normality", "groupwise_normality_tests.csv"),
    ("S4_group_comparisons", "group_comparisons.csv"),
    ("S5_posthoc_holm_tests", "posthoc_holm_tests.csv"),
    ("S6_spearman_correlations", "spearman_correlations.csv"),
    ("S7_cluster_sensitivity_6MWT", "cluster_sensitivity_6mwt.csv"),
    ("S8_cluster_sensitivity_climb", "cluster_sensitivity_simulated_climbing.csv"),
    ("S9_fatigue_association", "fatigue_index_association.csv"),
    ("S10_fatigue_tertiles", "fatigue_index_by_performance_tertile.csv"),
    ("S11_fatigue_profile_tests", "fatigue_index_by_profile_kruskal.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=None,
        help="Directory containing revised analysis CSV files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/outputs/revision_tables.",
    )
    return parser.parse_args()


def safe_sheet_name(name: str) -> str:
    invalid = set("[]:*?/\\")
    cleaned = "".join(ch for ch in name if ch not in invalid)
    return cleaned[:31]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stats_dir = args.stats_dir.resolve() if args.stats_dir else root / "outputs" / "revision_statistics"
    out_dir = args.out.resolve() if args.out else root / "outputs" / "revision_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    workbook = out_dir / "revision_supplementary_statistical_tables.xlsx"

    available_tables: list[tuple[str, Path, pd.DataFrame]] = []
    for table_name, file_name in TABLE_SPECS:
        path = stats_dir / file_name
        exists = path.exists()
        manifest_records.append(
            {
                "table": table_name,
                "source_file": str(path),
                "included": bool(exists),
            }
        )
        if exists:
            df = pd.read_csv(path)
            available_tables.append((table_name, path, df))

    manifest = pd.DataFrame(manifest_records)
    manifest.to_csv(out_dir / "supplementary_table_manifest.csv", index=False)

    if not available_tables:
        print(f"No CSV tables found in {stats_dir}. Wrote manifest only.")
        return

    try:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            manifest.to_excel(writer, sheet_name="Manifest", index=False)
            for table_name, _, df in available_tables:
                df.to_excel(writer, sheet_name=safe_sheet_name(table_name), index=False)
        print(f"Wrote supplementary statistical workbook to: {workbook}")
    except Exception as exc:  # pragma: no cover - fallback for limited environments.
        note = (
            f"Could not write Excel workbook because: {exc}\n"
            "The source CSV files remain available in outputs/revision_statistics/."
        )
        (out_dir / "excel_export_note.txt").write_text(note, encoding="utf-8")
        print(note)


if __name__ == "__main__":
    main()
