#!/usr/bin/env python3
"""Revised statistical analyses for the Scientific Reports revision.

This script reproduces the strengthened statistical workflow added during
revision: assumption checks, correlations, group-wise comparisons, effect
sizes, and Holm-adjusted exploratory post hoc tests.

Run from the repository root:

    python analysis/revision_statistical_analysis.py

or from another location:

    python analysis/revision_statistical_analysis.py --root /path/to/repository

Expected input files at the repository root:
    6MWT.txt
    Balance_Test.txt
    Balance_test_Mean.txt
    Simulated_climbing.txt
    SL.txt

Outputs are written to:
    outputs/revision_statistics/
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the source data files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/outputs/revision_statistics.",
    )
    return parser.parse_args()


def numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_whitespace_table(path: Path, names: list[str], min_cols: int) -> pd.DataFrame:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= min_cols:
                rows.append(parts[: len(names)])
    return pd.DataFrame(rows, columns=names)


def load_6mwt(root: Path) -> pd.DataFrame:
    names = [
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
    ]
    df = read_whitespace_table(root / "6MWT.txt", names, min_cols=14)
    numeric_cols = [c for c in names if c not in {"masked_name"}]
    df = numeric_frame(df, numeric_cols)
    df["body_weight_kg"] = df["body_weight_raw"] / 2.0
    df["bmi_kg_m2"] = df["body_weight_kg"] / (df["height_m"] ** 2)
    df["hr_change_bpm"] = df["post_hr_bpm"] - df["pre_hr_bpm"]
    df["sbp_change_mmhg"] = df["post_sbp_mmhg"] - df["pre_sbp_mmhg"]
    df["dbp_change_mmhg"] = df["post_dbp_mmhg"] - df["pre_dbp_mmhg"]
    df["spo2_change_pct"] = df["post_spo2_pct"] - df["pre_spo2_pct"]
    feature_cols = [
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
    df["cluster_6mwt_k4"] = kmeans_labels(df, feature_cols, 4)
    return df


def load_simulated_climbing(root: Path) -> pd.DataFrame:
    names = [
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
    ]
    df = read_whitespace_table(root / "Simulated_climbing.txt", names, min_cols=14)
    numeric_cols = [c for c in names if c not in {"masked_name"}]
    df = numeric_frame(df, numeric_cols)
    df["body_weight_kg"] = df["body_weight_raw"] / 2.0
    df["bmi_kg_m2"] = df["body_weight_kg"] / (df["height_m"] ** 2)
    df["hr_change_bpm"] = df["post_hr_bpm"] - df["pre_hr_bpm"]
    df["sbp_change_mmhg"] = df["post_sbp_mmhg"] - df["pre_sbp_mmhg"]
    df["dbp_change_mmhg"] = df["post_dbp_mmhg"] - df["pre_dbp_mmhg"]
    df["spo2_change_pct"] = df["post_spo2_pct"] - df["pre_spo2_pct"]
    feature_cols = [
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
    df["cluster_climbing_k4"] = kmeans_labels(df, feature_cols, 4)
    return df


def load_balance(root: Path) -> pd.DataFrame:
    names = [
        "participant_id",
        "masked_name",
        "age_y",
        "height_m",
        "body_weight_raw",
        "balance_trial1_s",
        "balance_trial2_s",
    ]
    df = read_whitespace_table(root / "Balance_Test.txt", names, min_cols=7)
    numeric_cols = [c for c in names if c not in {"masked_name"}]
    df = numeric_frame(df, numeric_cols)
    df["body_weight_kg"] = df["body_weight_raw"] / 2.0
    df["bmi_kg_m2"] = df["body_weight_kg"] / (df["height_m"] ** 2)
    df["balance_mean_s"] = df[["balance_trial1_s", "balance_trial2_s"]].mean(axis=1)
    df["balance_max_s"] = df[["balance_trial1_s", "balance_trial2_s"]].max(axis=1)
    return df


def load_squat_lift(root: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    path = root / "SL.txt"
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                record = {
                    "participant_id": float(parts[0]),
                    "age_y": float(parts[1]),
                    "height_m": float(parts[2]),
                    "body_weight_raw": float(parts[3]),
                    "repetitions": float(parts[4]),
                }
                if record["height_m"] > 10:
                    record["height_m"] = record["height_m"] / 100.0
                record["body_weight_kg"] = record["body_weight_raw"] / 2.0
                record["bmi_kg_m2"] = record["body_weight_kg"] / (record["height_m"] ** 2)
                record["emg_cluster"] = float(parts[5]) if len(parts) >= 6 else np.nan
                rows.append(record)
            except ValueError:
                continue
    return pd.DataFrame(rows)


def kmeans_labels(df: pd.DataFrame, feature_cols: list[str], n_clusters: int) -> np.ndarray:
    valid = df[feature_cols].dropna()
    if len(valid) < n_clusters:
        return np.full(len(df), np.nan)
    scaled = StandardScaler().fit_transform(valid)
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20).fit_predict(scaled)
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    result.loc[valid.index] = labels
    return result.to_numpy()


def shapiro_record(dataset: str, variable: str, values: pd.Series) -> dict[str, float | str]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return {"dataset": dataset, "variable": variable, "n": len(clean), "w": np.nan, "p": np.nan}
    w, p = stats.shapiro(clean)
    return {"dataset": dataset, "variable": variable, "n": len(clean), "w": w, "p": p}


def descriptive_records(dataset: str, df: pd.DataFrame, variables: list[str]) -> list[dict[str, float | str]]:
    records = []
    for variable in variables:
        clean = pd.to_numeric(df[variable], errors="coerce").dropna()
        records.append(
            {
                "dataset": dataset,
                "variable": variable,
                "n": len(clean),
                "mean": clean.mean(),
                "sd": clean.std(ddof=1),
                "median": clean.median(),
                "iqr": clean.quantile(0.75) - clean.quantile(0.25),
                "min": clean.min(),
                "max": clean.max(),
            }
        )
    return records


def spearman_records(
    dataset: str,
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> list[dict[str, float | str]]:
    records = []
    for predictor in predictors:
        pair = df[[outcome, predictor]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(pair[predictor], pair[outcome])
        records.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "predictor": predictor,
                "n": len(pair),
                "spearman_rho": rho,
                "p": p,
            }
        )
    return records


def eta_squared_anova(df: pd.DataFrame, value_col: str, group_col: str) -> float:
    data = df[[value_col, group_col]].dropna()
    grand_mean = data[value_col].mean()
    ss_between = sum(
        len(group) * (group[value_col].mean() - grand_mean) ** 2
        for _, group in data.groupby(group_col)
    )
    ss_total = sum((data[value_col] - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total else np.nan


def epsilon_squared_kruskal(h_stat: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return max(0.0, float((h_stat - k + 1) / (n - k)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p values in the original order."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        adj = min(1.0, (m - rank) * raw)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def group_comparison(
    dataset: str,
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> tuple[dict[str, float | str], list[dict[str, float | str]], list[dict[str, float | str]]]:
    data = df[[value_col, group_col]].apply(pd.to_numeric, errors="coerce").dropna()
    groups = [g[value_col].dropna().to_numpy() for _, g in data.groupby(group_col)]
    group_labels = [str(label) for label, _ in data.groupby(group_col)]
    k = len(groups)
    n = sum(len(g) for g in groups)

    normality = []
    for label, values in zip(group_labels, groups):
        if len(values) >= 3:
            w, p = stats.shapiro(values)
        else:
            w, p = np.nan, np.nan
        normality.append(
            {
                "dataset": dataset,
                "variable": value_col,
                "group_variable": group_col,
                "group": label,
                "n": len(values),
                "shapiro_w": w,
                "shapiro_p": p,
            }
        )

    if k < 2 or n <= k:
        comparison = {
            "dataset": dataset,
            "variable": value_col,
            "group_variable": group_col,
            "n": n,
            "groups": k,
            "test": "not_tested",
            "statistic": np.nan,
            "p": np.nan,
            "effect_size_name": "",
            "effect_size": np.nan,
            "levene_statistic": np.nan,
            "levene_p": np.nan,
        }
        return comparison, normality, []

    levene_stat, levene_p = stats.levene(*groups, center="median")
    normal_ok = all(len(g) >= 3 for g in groups) and all(
        (rec["shapiro_p"] > ALPHA) for rec in normality if not pd.isna(rec["shapiro_p"])
    )
    variance_ok = bool(levene_p > ALPHA)

    if normal_ok and variance_ok:
        stat, p = stats.f_oneway(*groups)
        test = "one_way_anova"
        effect_name = "eta_squared"
        effect = eta_squared_anova(data, value_col, group_col)
    else:
        stat, p = stats.kruskal(*groups)
        test = "kruskal_wallis"
        effect_name = "epsilon_squared"
        effect = epsilon_squared_kruskal(stat, n, k)

    comparison = {
        "dataset": dataset,
        "variable": value_col,
        "group_variable": group_col,
        "n": n,
        "groups": k,
        "test": test,
        "statistic": stat,
        "p": p,
        "effect_size_name": effect_name,
        "effect_size": effect,
        "levene_statistic": levene_stat,
        "levene_p": levene_p,
    }

    posthoc_records: list[dict[str, float | str]] = []
    if p < ALPHA and k > 2:
        pair_records = []
        pair_p_values = []
        grouped = {str(label): group[value_col].dropna().to_numpy() for label, group in data.groupby(group_col)}
        for g1, g2 in itertools.combinations(grouped.keys(), 2):
            x, y = grouped[g1], grouped[g2]
            if len(x) < 1 or len(y) < 1:
                stat_pair, p_pair = np.nan, np.nan
                method = "not_tested"
            elif test == "kruskal_wallis":
                stat_pair, p_pair = stats.mannwhitneyu(x, y, alternative="two-sided")
                method = "mann_whitney_u"
            else:
                stat_pair, p_pair = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                method = "welch_t_test"
            pair_records.append(
                {
                    "dataset": dataset,
                    "variable": value_col,
                    "group_variable": group_col,
                    "group_1": g1,
                    "group_2": g2,
                    "method": method,
                    "statistic": stat_pair,
                    "p_raw": p_pair,
                }
            )
            pair_p_values.append(p_pair)
        adjusted = holm_adjust([p for p in pair_p_values if not pd.isna(p)])
        adj_iter = iter(adjusted)
        for rec in pair_records:
            rec["p_holm"] = next(adj_iter) if not pd.isna(rec["p_raw"]) else np.nan
            posthoc_records.append(rec)

    return comparison, normality, posthoc_records


def write_outputs(out_dir: Path) -> None:
    root = out_dir.parent.parent
    six = load_6mwt(root)
    sim = load_simulated_climbing(root)
    bal = load_balance(root)
    sl = load_squat_lift(root)

    out_dir.mkdir(parents=True, exist_ok=True)

    descriptive = []
    descriptive += descriptive_records(
        "6MWT",
        six,
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2", "laps", "pre_hr_bpm", "post_hr_bpm"],
    )
    descriptive += descriptive_records(
        "Simulated climbing",
        sim,
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2", "climb_time_s", "pre_hr_bpm", "post_hr_bpm"],
    )
    descriptive += descriptive_records(
        "Balance",
        bal,
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2", "balance_mean_s", "balance_max_s"],
    )
    descriptive += descriptive_records(
        "Squat-and-lift",
        sl,
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2", "repetitions"],
    )
    pd.DataFrame(descriptive).to_csv(out_dir / "descriptive_statistics.csv", index=False)

    normality = []
    for dataset, frame, variables in [
        ("6MWT", six, ["laps", "age_y", "height_m", "body_weight_kg", "bmi_kg_m2"]),
        ("Simulated climbing", sim, ["climb_time_s", "age_y", "height_m", "body_weight_kg", "bmi_kg_m2"]),
        ("Balance", bal, ["balance_mean_s", "balance_max_s"]),
        ("Squat-and-lift", sl, ["repetitions", "age_y", "body_weight_kg", "bmi_kg_m2"]),
    ]:
        for variable in variables:
            normality.append(shapiro_record(dataset, variable, frame[variable]))
    pd.DataFrame(normality).to_csv(out_dir / "normality_tests.csv", index=False)

    correlations = []
    correlations += spearman_records(
        "6MWT",
        six,
        "laps",
        [
            "age_y",
            "height_m",
            "body_weight_kg",
            "bmi_kg_m2",
            "pre_sbp_mmhg",
            "pre_dbp_mmhg",
            "pre_spo2_pct",
            "pre_hr_bpm",
            "hr_change_bpm",
        ],
    )
    correlations += spearman_records(
        "Simulated climbing",
        sim,
        "climb_time_s",
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2", "pre_hr_bpm", "hr_change_bpm"],
    )
    correlations += spearman_records(
        "Balance",
        bal,
        "balance_max_s",
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2"],
    )
    correlations += spearman_records(
        "Squat-and-lift",
        sl,
        "repetitions",
        ["age_y", "height_m", "body_weight_kg", "bmi_kg_m2"],
    )
    pd.DataFrame(correlations).to_csv(out_dir / "spearman_correlations.csv", index=False)

    comparisons = []
    group_normality = []
    posthoc = []
    comparison_specs = [
        ("6MWT", six, "laps", "cluster_6mwt_k4"),
        ("6MWT", six, "bmi_kg_m2", "cluster_6mwt_k4"),
        ("6MWT", six, "age_y", "cluster_6mwt_k4"),
        ("6MWT", six, "height_m", "cluster_6mwt_k4"),
        ("Simulated climbing", sim, "climb_time_s", "cluster_climbing_k4"),
        ("Simulated climbing", sim, "bmi_kg_m2", "cluster_climbing_k4"),
        ("Simulated climbing", sim, "height_m", "cluster_climbing_k4"),
        ("Squat-and-lift", sl, "repetitions", "emg_cluster"),
        ("Squat-and-lift", sl, "body_weight_kg", "emg_cluster"),
        ("Squat-and-lift", sl, "bmi_kg_m2", "emg_cluster"),
        ("Squat-and-lift", sl, "age_y", "emg_cluster"),
    ]
    for dataset, frame, value_col, group_col in comparison_specs:
        comp, norm, pairwise = group_comparison(dataset, frame, value_col, group_col)
        comparisons.append(comp)
        group_normality.extend(norm)
        posthoc.extend(pairwise)

    pd.DataFrame(comparisons).to_csv(out_dir / "group_comparisons.csv", index=False)
    pd.DataFrame(group_normality).to_csv(out_dir / "groupwise_normality_tests.csv", index=False)
    pd.DataFrame(posthoc).to_csv(out_dir / "posthoc_holm_tests.csv", index=False)

    print(f"Wrote revised statistical outputs to: {out_dir}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve() if args.out else root / "outputs" / "revision_statistics"
    # The output function infers root as <out>/../.. for the default layout.
    # For custom output directories, temporarily pass the default root-based path.
    if args.out is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        default_out = root / "outputs" / "revision_statistics"
        write_outputs(default_out)
        if out_dir != default_out:
            for file in default_out.glob("*.csv"):
                file.replace(out_dir / file.name)
    else:
        write_outputs(out_dir)


if __name__ == "__main__":
    main()
