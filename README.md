- # Sensor-Data-Based Assessment of Operational Capability in Transmission-Line Work at Height

  This repository contains the source data, subject-level files, and analysis scripts used in the study **“Sensor-Data-Based Assessment of Operational Capability in Transmission-Line Work at Height”** It is organized as a public supplementary repository to support editorial review, peer review, and result verification.

  ## Purpose of this repository

  The materials provided here are intended to help reviewers and readers understand how the analyses in the manuscript were carried out and how the main figures were generated. The repository includes:

  - source data used in the four experimental tasks;
  - subject-level files used in the squat-and-lift / EMG analysis;
  - core Python scripts used for data processing, clustering, curve fitting, and figure generation.

  This repository is not a polished software package. It is a research archive prepared to improve transparency and reproducibility.

  ## Study overview

  The manuscript evaluates fatigue-related physical capability in high-altitude transmission-line workers using four task-oriented tests:

  1. **6-minute walk test (6MWT)** for cardiorespiratory endurance;
  2. **squat-and-lift task** for muscular performance and EMG-based feature extraction;
  3. **eyes-closed single-leg stance test** for balance performance;
  4. **simulated climbing task** for integrated climbing-related operational performance.

  The analyses combine correlation analysis, distribution fitting, clustering, and multidimensional group comparison.

  ## Repository structure

  ### Top-level source data and scripts

  At the top level of the repository, the following files are provided:

  #### Raw / source data
  - `6MWT.txt`
  - `Balance_Test.txt`
  - `Balance_test_Mean.txt`
  - `Simulated_climbing.txt`
  - `Simulated_climbing_clustered_3d.txt`
  - `SL.txt`

  #### Core analysis scripts
  - `plotting_style.py`
  - `6MWT_Categorize.py`
  - `6MWT_combined_analysis.py`
  - `6MWT_group_analysis_with_fatigue.py`
  - `Balance_test_combined_analysis.py`
  - `Balance_test_distribution_grading.py`
  - `Simulated_climbing_cluster_combined.py`
  - `Simulated_climbing_group_analysis_with_fatigue.py`
  - `SL_MSG_fitting.py`
  - `SL_kmeans.py`
  - `SL_weight_combined_analysis.py`

  ## Software environment
  - Python 3.13
  - Main packages: numpy, pandas, matplotlib, scipy, scikit-learn

  ## Subject-level folders
  
  Folders named by participant or trial identifiers (for example `2`, `4`, `5`, ..., `36`, and `8（顺序变动，腰，肩，大腿，小腿）`) contain subject-level files used mainly in the squat-and-lift / EMG analysis.
  
  These folders typically include:
  - `MVC.txt`
  - `MVC.asc`
  - `MVC.avi`
  - `MVC_single_sheet.xlsx`
  - `global_standard_values.xlsx`
  - `detailed_results.xlsx`
  - `plots/`
  
  These files were used to:
  - extract EMG signal segments;
  - calculate reference values;
  - detect deviation intervals;
  - perform polynomial fitting;
  - construct feature matrices for clustering.
  
  ## Main analyses and corresponding scripts
  
  ### 1. 6MWT analysis
  Relevant files:
  - `6MWT.txt`
  - `6MWT_Categorize.py`
  - `6MWT_combined_analysis.py`
  - `6MWT_group_analysis_with_fatigue.py`
  
  Main outputs:
  - clustering of 6MWT participants;
  - correlation analysis between physiological variables and walking performance;
  - group-wise comparison of fitness indicators and fatigue scores.
  
  ### 2. Balance test analysis
  Relevant files:
  - `Balance_Test.txt`
  - `Balance_test_Mean.txt`
  - `Balance_test_combined_analysis.py`
  - `Balance_test_distribution_grading.py`
  
  Main outputs:
  - comparison of the two balance trials;
  - fitted distribution of averaged balance duration;
  - normal-distribution-based grading.
  
  ### 3. Simulated climbing analysis
  Relevant files:
  - `Simulated_climbing.txt`
  - `Simulated_climbing_clustered_3d.txt`
  - `Simulated_climbing_cluster_combined.py`
  - `Simulated_climbing_group_analysis_with_fatigue.py`
  
  Main outputs:
  - 2D/3D clustering visualization;
  - group-wise comparison of climbing-related indicators;
  - comparison with post-work fatigue.
  
  ### 4. Squat-and-lift / EMG analysis
  Relevant files:
  - subject-level folders (`2`, `4`, `5`, ..., `36`)
  - `SL.txt`
  - `SL_MSG_fitting.py`
  - `SL_kmeans.py`
  - `SL_weight_combined_analysis.py`
  
  Main outputs:
  - EMG-based interval extraction and polynomial fitting;
  - subject clustering based on EMG-derived features;
  - group-wise comparison of weight, repetitions, age, and BMI.

  ## Notes on reproducibility
  
  Some scripts were originally written to be run in a working directory where the corresponding data files are located in the same folder. If you plan to rerun the analyses, you may need to:
  
  - place the relevant `.txt` or subject folders in the same working directory as the script being executed; or
  - modify file paths inside the scripts to match your local environment.

  The scripts in this repository are the research versions used during analysis. They were prepared for reproducibility rather than for packaging as a general-purpose software release.

  ## Data and privacy

  The repository is intended for scientific review and reproducibility. The files included here are those necessary to understand and reproduce the main analyses presented in the manuscript. If any version of the repository is made publicly available beyond peer review, sensitive or personally identifiable information should be removed or anonymized before release.

  ## Manuscript correspondence

  This repository supports the manuscript:

  **Sensor-Data-Based Assessment of Operational Capability in Transmission-Line Work at Height**

  If needed, the manuscript file can be provided separately as part of the submission package.
  
  ## Contact
  
  For questions regarding the repository contents, data structure, or analysis scripts, please contact the corresponding author listed in the manuscript.
