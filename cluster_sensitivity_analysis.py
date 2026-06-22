#!/usr/bin/env python3
"""Cluster-solution sensitivity analysis for the revised manuscript.

The revised manuscript reports k = 2-5 sensitivity checks for the 6MWT and
simulated-climbing K-means profiles. This script reproduces those checks and
writes one CSV per task.

Run from the repository root:

    python analysis/cluster_sensitivity_analysis.py

Outputs are written to:
    outputs/revision_statistics/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing 6MWT.txt and Simulated_climbing.txt.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/outputs/revision_statistics.",
    )
    return parser.parse_args()


def read_numeric_matrix(path: Path, selected_indices: list[int], min_cols: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < min_cols:
                continue
            try:
                rows.append([float(parts[i]) for i in selected_indices])
            except (ValueError, IndexError):
                continue
    return np.asarray(rows, dtype=float)


def sensitivity_table(task: str, matrix: np.ndarray, k_values: range) -> pd.DataFrame:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"No usable rows were found for {task}.")

    scaled = StandardScaler().fit_transform(matrix)
    records = []
    for k in k_values:
        if scaled.shape[0] <= k:
            continue
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(scaled)
        unique, counts = np.unique(labels, return_counts=True)
        size_map = {int(label): int(count) for label, count in zip(unique, counts)}

        records.append(
            {
                "task": task,
                "k": k,
                "n": int(scaled.shape[0]),
                "silhouette": silhouette_score(scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(scaled, labels),
                "davies_bouldin": davies_bouldin_score(scaled, labels),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "cluster_sizes": "; ".join(f"{label}:{size_map[label]}" for label in sorted(size_map)),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve() if args.out else root / "outputs" / "revision_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # These selected columns follow the original analysis scripts:
    # 6MWT_Categorize.py used columns 3:12.
    six_matrix = read_numeric_matrix(
        root / "6MWT.txt",
        selected_indices=list(range(3, 12)),
        min_cols=14,
    )
    # Simulated_climbing_cluster_combined.py used columns 2:12.
    climbing_matrix = read_numeric_matrix(
        root / "Simulated_climbing.txt",
        selected_indices=list(range(2, 12)),
        min_cols=14,
    )

    six_table = sensitivity_table("6MWT", six_matrix, range(2, 6))
    climb_table = sensitivity_table("Simulated climbing", climbing_matrix, range(2, 6))

    six_table.to_csv(out_dir / "cluster_sensitivity_6mwt.csv", index=False)
    climb_table.to_csv(out_dir / "cluster_sensitivity_simulated_climbing.csv", index=False)

    combined = pd.concat([six_table, climb_table], ignore_index=True)
    combined.to_csv(out_dir / "cluster_sensitivity_combined.csv", index=False)

    print(f"Wrote cluster-sensitivity outputs to: {out_dir}")


if __name__ == "__main__":
    main()
