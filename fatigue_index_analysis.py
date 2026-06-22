#!/usr/bin/env python3
"""Fatigue-index analyses added during revision.

This script builds an exploratory objective task-performance index from the
four task outcomes and, when a de-identified fatigue file is present, tests
its association with self-reported fatigue index.

Expected task files:
    6MWT.txt
    Balance_Test.txt
    Simulated_climbing.txt
    SL.txt

Optional fatigue file:
    fatigue_index.csv
or:
    data/fatigue_index.csv

The fatigue file should contain one participant identifier column and one
fatigue-index column. Accepted column names include:
    participant_id, id, subject_id
    fatigue_index, fatigue, fatigue_score, post_work_fatigue

Outputs are written to:
    outputs/revision_statistics/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/outputs/revision_statistics.",
    )
    parser.add_argument(
        "--require-fatigue-data",
        action="store_true",
        help="Return an error if no de-identified fatigue_index.csv is found.",
    )
    return parser.parse_args()


def read_whitespace(path: Path, names: list[str], min_cols: int) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= min_cols:
                rows.append(parts[: len(names)])
    df = pd.DataFrame(rows, columns=names)
    for col in df.columns:
        if col != "masked_name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def zscore(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not higher_is_better:
        numeric = -numeric
    sd = numeric.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (numeric - numeric.mean()) / sd


def kmeans_labels(df: pd.DataFrame, feature_cols: list[str], n_clusters: int) -> pd.Series:
    valid = df[feature_cols].dropna()
    labels = pd.Series(np.nan, index=df.index, dtype="float64")
    if len(valid) < n_clusters:
        return labels
    scaled = StandardScaler().fit_transform(valid)
    labels.loc[valid.index] = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit_predict(scaled)
    return labels


def load_task_performance(root: Path) -> pd.DataFrame:
    six = read_whitespace(
        root / "6MWT.txt",
        [
            "participant_id",
            "masked_name",
            "age_y",
            "height_m",
            "body_weight_raw",
            "pre_sbp_mmhg",
            "pre_dbp_mmhg",
            "post_sbp_mmhg",
            "post_dbp_mmhg",
            "pre_spo2_pct",
            "post_spo2_pct",
            "pre_hr_bpm",
            "post_hr_bpm",
            "laps",
        ],
        min_cols=14,
    )
    six_features = [
        "height_m",
        "body_weight_raw",
        "pre_sbp_mmhg",
        "pre_dbp_mmhg",
        "post_sbp_mmhg",
        "post_dbp_mmhg",
        "pre_spo2_pct",
        "post_spo2_pct",
        "pre_hr_bpm",
    ]
    six["cluster_6mwt_k4"] = kmeans_labels(six, six_features, 4)
    six = six[["participant_id", "laps", "cluster_6mwt_k4"]]

    balance = read_whitespace(
        root / "Balance_Test.txt",
        [
            "participant_id",
            "masked_name",
            "age_y",
            "height_m",
            "body_weight_raw",
            "balance_trial1_s",
            "balance_trial2_s",
        ],
        min_cols=7,
    )
    balance["balance_max_s"] = balance[["balance_trial1_s", "balance_trial2_s"]].max(axis=1)
    balance = balance[["participant_id", "balance_max_s"]]

    climbing = read_whitespace(
        root / "Simulated_climbing.txt",
        [
            "participant_id",
            "masked_name",
            "age_y",
            "height_m",
            "body_weight_raw",
            "pre_sbp_mmhg",
            "pre_dbp_mmhg",
            "post_sbp_mmhg",
            "post_dbp_mmhg",
            "pre_hr_bpm",
            "post_hr_bpm",
            "pre_spo2_pct",
            "post_spo2_pct",
            "climb_time_s",
        ],
        min_cols=14,
    )
    climbing_features = [
        "age_y",
        "height_m",
        "body_weight_raw",
        "pre_sbp_mmhg",
        "pre_dbp_mmhg",
        "post_sbp_mmhg",
        "post_dbp_mmhg",
        "pre_hr_bpm",
        "post_hr_bpm",
        "pre_spo2_pct",
    ]
    climbing["cluster_climbing_k4"] = kmeans_labels(climbing, climbing_features, 4)
    climbing = climbing[["participant_id", "climb_time_s", "cluster_climbing_k4"]]

    squat_rows = []
    with (root / "SL.txt").open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                squat_rows.append({"participant_id": float(parts[0]), "repetitions": float(parts[4])})
            except ValueError:
                continue
    squat = pd.DataFrame(squat_rows)

    perf = six.merge(balance, on="participant_id", how="outer")
    perf = perf.merge(climbing, on="participant_id", how="outer")
    perf = perf.merge(squat, on="participant_id", how="outer")
    perf["score_6mwt_laps_z"] = zscore(perf["laps"], higher_is_better=True)
    perf["score_balance_max_z"] = zscore(perf["balance_max_s"], higher_is_better=True)
    perf["score_squat_lift_repetitions_z"] = zscore(perf["repetitions"], higher_is_better=True)
    perf["score_climb_time_z"] = zscore(perf["climb_time_s"], higher_is_better=False)
    score_cols = [
        "score_6mwt_laps_z",
        "score_balance_max_z",
        "score_squat_lift_repetitions_z",
        "score_climb_time_z",
    ]
    perf["available_task_scores"] = perf[score_cols].notna().sum(axis=1)
    perf["objective_task_performance_index"] = perf[score_cols].mean(axis=1, skipna=True)
    return perf.sort_values("participant_id")


def normalize_column_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum() or ch == "_")


def load_fatigue_index(root: Path) -> pd.DataFrame | None:
    candidates = [
        root / "fatigue_index.csv",
        root / "data" / "fatigue_index.csv",
        root / "source_data" / "fatigue_index.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        lookup = {normalize_column_name(col): col for col in df.columns}
        id_col = None
        fatigue_col = None
        for key in ["participant_id", "participantid", "id", "subject_id", "subjectid", "worker_id"]:
            if key in lookup:
                id_col = lookup[key]
                break
        for key in ["fatigue_index", "fatigueindex", "fatigue_score", "fatiguescore", "fatigue", "post_work_fatigue"]:
            if key in lookup:
                fatigue_col = lookup[key]
                break
        if id_col is None or fatigue_col is None:
            raise ValueError(
                f"{path} was found, but it does not contain recognizable participant and fatigue columns."
            )
        result = df[[id_col, fatigue_col]].rename(
            columns={id_col: "participant_id", fatigue_col: "fatigue_index"}
        )
        result["participant_id"] = pd.to_numeric(result["participant_id"], errors="coerce")
        result["fatigue_index"] = pd.to_numeric(result["fatigue_index"], errors="coerce")
        return result.dropna(subset=["participant_id", "fatigue_index"])
    return None


def spearman_result(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> dict[str, float | str]:
    data = df[[x_col, y_col]].dropna()
    if len(data) < 3:
        rho, p = np.nan, np.nan
    else:
        rho, p = stats.spearmanr(data[x_col], data[y_col])
    return {
        "analysis": label,
        "x": x_col,
        "y": y_col,
        "n": len(data),
        "spearman_rho": rho,
        "p": p,
    }


def kruskal_by_group(df: pd.DataFrame, value_col: str, group_col: str, label: str) -> dict[str, float | str]:
    data = df[[value_col, group_col]].dropna()
    groups = [group[value_col].to_numpy() for _, group in data.groupby(group_col)]
    if len(groups) < 2:
        h, p = np.nan, np.nan
    else:
        h, p = stats.kruskal(*groups)
    return {
        "analysis": label,
        "value": value_col,
        "group": group_col,
        "n": len(data),
        "groups": len(groups),
        "kruskal_h": h,
        "p": p,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve() if args.out else root / "outputs" / "revision_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    performance = load_task_performance(root)
    performance.to_csv(out_dir / "objective_task_performance_index.csv", index=False)

    fatigue = load_fatigue_index(root)
    if fatigue is None:
        message = (
            "No de-identified fatigue_index.csv was found. The objective task-performance "
            "index was generated, but fatigue-index correlations and Kruskal-Wallis "
            "comparisons were not run. Add fatigue_index.csv with columns participant_id "
            "and fatigue_index to reproduce the fatigue analyses."
        )
        (out_dir / "fatigue_index_missing_data_note.txt").write_text(message, encoding="utf-8")
        if args.require_fatigue_data:
            raise FileNotFoundError(message)
        print(message)
        return

    merged = performance.merge(fatigue, on="participant_id", how="inner")
    merged.to_csv(out_dir / "fatigue_index_analysis_dataset.csv", index=False)

    associations = [
        spearman_result(
            merged,
            "objective_task_performance_index",
            "fatigue_index",
            "Objective task-performance index vs fatigue index",
        ),
        spearman_result(merged, "climb_time_s", "fatigue_index", "Climb time vs fatigue index"),
        spearman_result(merged, "laps", "fatigue_index", "6MWT laps vs fatigue index"),
        spearman_result(merged, "balance_max_s", "fatigue_index", "Balance max duration vs fatigue index"),
        spearman_result(merged, "repetitions", "fatigue_index", "Squat-and-lift repetitions vs fatigue index"),
    ]
    pd.DataFrame(associations).to_csv(out_dir / "fatigue_index_association.csv", index=False)

    tertile_data = merged.dropna(subset=["objective_task_performance_index", "fatigue_index"]).copy()
    if len(tertile_data) >= 6:
        tertile_data["performance_tertile"] = pd.qcut(
            tertile_data["objective_task_performance_index"],
            q=3,
            labels=["lower", "middle", "higher"],
            duplicates="drop",
        )
        tertile_summary = (
            tertile_data.groupby("performance_tertile", observed=True)["fatigue_index"]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        tertile_summary.to_csv(out_dir / "fatigue_index_by_performance_tertile.csv", index=False)
        kw_tertile = kruskal_by_group(
            tertile_data,
            "fatigue_index",
            "performance_tertile",
            "Fatigue index by objective performance tertile",
        )
        pd.DataFrame([kw_tertile]).to_csv(out_dir / "fatigue_index_tertile_kruskal.csv", index=False)

    profile_tests = [
        kruskal_by_group(merged, "fatigue_index", "cluster_6mwt_k4", "Fatigue index by 6MWT profile"),
        kruskal_by_group(
            merged,
            "fatigue_index",
            "cluster_climbing_k4",
            "Fatigue index by simulated-climbing profile",
        ),
    ]
    pd.DataFrame(profile_tests).to_csv(out_dir / "fatigue_index_by_profile_kruskal.csv", index=False)

    print(f"Wrote fatigue-index outputs to: {out_dir}")


if __name__ == "__main__":
    main()
