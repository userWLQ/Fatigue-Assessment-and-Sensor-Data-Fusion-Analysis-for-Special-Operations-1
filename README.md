# Sensor-Data-Based Assessment of Operational Capability in Transmission-Line Work at Height

This repository supports the revised manuscript:

**Sensor-Data-Based Assessment of Operational Capability in Transmission-Line Work at Height**

The repository contains task-level data, subject-level EMG files, original analysis scripts, and revised statistical analysis scripts used to support the exploratory multisource assessment framework reported in the manuscript and Supplementary Information.

This repository is intended as a research archive for editorial review, peer review, and result verification. It is not a general-purpose software package.

## Study overview

The study assessed operational capability in overhead transmission-line workers performing work-at-height-related tasks. Four standardized tests were analyzed:

1. **6-minute walk test (6MWT)** for cardiorespiratory performance.
2. **Squat-and-lift task** for muscular performance and EMG-derived signal profiles.
3. **Eyes-closed single-leg stance test** for balance performance.
4. **Simulated climbing task** for climbing-related operational performance.

The revised manuscript treats the proposed profiles as exploratory and descriptive. The analysis combines correlation analysis, distribution-based grouping, K-means clustering, cluster-validity metrics, sensitivity analysis, group-wise statistical comparisons, and supportive comparison with self-reported fatigue index.

## Repository contents

### Task-level source data

The following task-level files are provided at the repository root:

```text
6MWT.txt
Balance_Test.txt
Balance_test_Mean.txt
Simulated_climbing.txt
Simulated_climbing_clustered_3d.txt
SL.txt
```

These files support the 6MWT, balance, simulated-climbing, and squat-and-lift analyses.

### Original analysis scripts

The original analysis scripts are retained at the repository root:

```text
plotting_style.py
6MWT_Categorize.py
6MWT_combined_analysis.py
6MWT_group_analysis_with_fatigue.py
Balance_test_combined_analysis.py
Balance_test_distribution_grading.py
Simulated_climbing_cluster_combined.py
Simulated_climbing_group_analysis_with_fatigue.py
SL_MSG_fitting.py
SL_kmeans.py
SL_weight_combined_analysis.py
```

These scripts were used for the original data processing, clustering, curve fitting, EMG feature preparation, and figure generation.

### Revised analysis scripts

The revised submission adds a cleaner analysis entry point in the `analysis/` folder:

```text
analysis/
|-- revision_statistical_analysis.py
|-- cluster_sensitivity_analysis.py
|-- fatigue_index_analysis.py
|-- supplementary_tables_generation.py
`-- run_all_revision_analyses.py
```

These scripts reproduce the analyses added or strengthened during revision:

- Shapiro-Wilk normality tests
- Levene homogeneity-of-variance tests
- Spearman rank correlations
- ANOVA or Kruskal-Wallis group comparisons
- Effect-size calculation
- Holm-adjusted exploratory pairwise tests
- 6MWT clustering sensitivity analysis for k = 2-5
- Simulated-climbing clustering sensitivity analysis for k = 2-5
- Exploratory fatigue-index association analysis
- Supplementary statistical table generation

### Subject-level EMG folders

Folders named by participant or trial identifiers, such as `2`, `4`, `5`, ..., `36`, contain subject-level files used mainly in the squat-and-lift and EMG analysis. These folders may include:

```text
Test.asc
MVC.asc
MVC.txt
MVC_single_sheet.xlsx
global_standard_values.xlsx
detailed_results.xlsx
plots/
```

These files were used to extract EMG signal segments, calculate reference values, identify signal intervals, perform polynomial fitting, and construct feature matrices for clustering.

Some subject-level folders contain files named `MVC.*`. These filenames reflect the original acquisition/export structure. The revised manuscript did not use formal MVC normalization for the reported EMG-derived profiles.

Video recordings are not included in the public reproducibility package because they may contain directly identifiable participant or workplace information. The public repository is intended to retain de-identified numerical signals, processed outputs, and analysis scripts only.

## Software environment

The revised analysis scripts require Python and the packages listed in `requirements.txt`.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The main packages are:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
openpyxl
```

## Running the revised analyses

From the repository root, run all revised analyses with:

```bash
python analysis/run_all_revision_analyses.py
```

The full workflow runs:

```text
revision_statistical_analysis.py
cluster_sensitivity_analysis.py
fatigue_index_analysis.py
supplementary_tables_generation.py
```

Each script can also be run separately:

```bash
python analysis/revision_statistical_analysis.py
python analysis/cluster_sensitivity_analysis.py
python analysis/fatigue_index_analysis.py
python analysis/supplementary_tables_generation.py
```

If a de-identified fatigue-index file is included as `fatigue_index.csv` or `data/fatigue_index.csv`, the fatigue-index association analyses will also be run. If this file is absent, the script still generates the objective task-performance index and writes a note indicating that fatigue-index correlations were not run.

## Revised analysis outputs

The revised scripts write statistical outputs to:

```text
outputs/revision_statistics/
```

Key output files include:

```text
descriptive_statistics.csv
normality_tests.csv
groupwise_normality_tests.csv
group_comparisons.csv
posthoc_holm_tests.csv
spearman_correlations.csv
cluster_sensitivity_6mwt.csv
cluster_sensitivity_simulated_climbing.csv
objective_task_performance_index.csv
fatigue_index_association.csv
fatigue_index_by_performance_tertile.csv
fatigue_index_by_profile_kruskal.csv
```

The supplementary table generator writes:

```text
outputs/revision_tables/revision_supplementary_statistical_tables.xlsx
outputs/revision_tables/supplementary_table_manifest.csv
```

## Main analysis mapping

### 6MWT analysis

Relevant files:

```text
6MWT.txt
6MWT_Categorize.py
6MWT_combined_analysis.py
6MWT_group_analysis_with_fatigue.py
analysis/revision_statistical_analysis.py
analysis/cluster_sensitivity_analysis.py
```

Main outputs include correlation analysis, K-means profiling, group-wise comparisons, and k = 2-5 sensitivity analysis.

### Balance test analysis

Relevant files:

```text
Balance_Test.txt
Balance_test_Mean.txt
Balance_test_combined_analysis.py
Balance_test_distribution_grading.py
analysis/revision_statistical_analysis.py
```

Main outputs include repeated-trial summaries, distribution-based grouping, and descriptive/statistical summaries of balance performance.

### Simulated climbing analysis

Relevant files:

```text
Simulated_climbing.txt
Simulated_climbing_clustered_3d.txt
Simulated_climbing_cluster_combined.py
Simulated_climbing_group_analysis_with_fatigue.py
analysis/revision_statistical_analysis.py
analysis/cluster_sensitivity_analysis.py
```

Main outputs include climbing-performance clustering, PCA visualization, group-wise comparison, and k = 2-5 sensitivity analysis.

### Squat-and-lift / EMG analysis

Relevant files:

```text
SL.txt
SL_MSG_fitting.py
SL_kmeans.py
SL_weight_combined_analysis.py
subject-level folders
analysis/revision_statistical_analysis.py
```

Main outputs include EMG-based feature preparation, cosine-similarity-based clustering, and group-wise comparison of repetitions, body weight, BMI, and age.

### Fatigue-index analysis

Relevant files:

```text
analysis/fatigue_index_analysis.py
fatigue_index.csv
```

The fatigue-index analysis is supportive and exploratory. Self-reported fatigue is treated as subjective convergent information, not as external validation of field performance, task failure, supervisor-rated capability, or safety outcomes.

## Notes on interpretation

The clustering analyses are exploratory. The four-level profile structure was retained in the revised manuscript to keep the task-specific grading framework consistent across tests, but the sensitivity analyses should be used to judge profile stability and separation.

The revised scripts are designed to improve transparency of the statistical workflow. They should be interpreted together with the Methods, Results, Discussion, and Supplementary Information of the revised manuscript.

## Data and privacy

This repository is intended for scientific review and reproducibility. The publicly shared version contains de-identified numerical data and analysis files. Direct personal identifiers, individual health notes, and identifiable video content should not be included in the public repository. Any controlled-access materials, if needed, should be shared only under the approved ethics and consent conditions.

## Data availability statement

The de-identified raw and processed data supporting the findings of this study, together with the analysis files used for sensor-data processing and statistical analysis, are available at:

https://github.com/userWLQ/Fatigue-Assessment-and-Sensor-Data-Fusion-Analysis-for-Special-Operations-1

## Contact

For questions about the repository contents, data structure, or analysis scripts, please contact the corresponding author listed in the manuscript.
