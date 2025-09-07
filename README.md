# TripleBranchNet
This repository contains a series of Python scripts to download, preprocess, analyze, train, and evaluate a machine learning model for the purpose of detecting crosstalk induced fidelity degredation in MNISQ based circuits.

## How to cite

TripleBranchNet is free-to-use and to modify. However, you must add reference [1] in case of any publication/production/execution/etc.

[1] to be updated later.


# Directory Structure
The project is organized into a sequential workflow, with each script performing a specific step. The scripts expect a relative directory structure to maintain portability.

```text
project_directory/
├── step_01_load_mnisq_dataset.py
├── step_02_create_train_test_dataset_sample.py
├── step_03_create_derivative_dataset_for_simulation.py
├── step_04_clean_and_check_derivative_dataset.py
├── step_05_manual_train_test_val_split_for_model_training.py
├── step_06_create_standard_scalers_for_dataset.py
├── step_07_train_tripleBranchNet_model.py
├── step_08_create_figures_from_training_logs.py
└── step_09_evaluate_trained_tripleBranchNet.py
├── mnisq_cache/
│   ├── archives/
│   │   ├── base_test_mnist_784_f80.zip
│   │   └── ... (all downloaded original zip files)
│   ├── base_test_mnist_784_f80/
│   │   ├── qasm/
│   │   ├── state/
│   │   ├── label/
│   │   └── fidelity/
│   ├── ... (other extracted original dataset folders, e.g., base_train_orig_mnist_784_f90)
│   └── download_log.log
├── combined_datasets/
│   └── combined_mixed_YYYYMMDD_HHMMSS_COMBINED/  (e.g., combined_mixed_20250617_140000_COMBINED)
│       ├── qasm/
│       │   ├── 00000_1_mixed_f80_test.json
│       │   └── ... (sampled QASM files from various sources)
│       ├── state/
│       │   ├── 00000_1_mixed_f80_test.npy  (Note: Assuming .npy, but original code used .json which is unusual)
│       │   └── ... (corresponding state files)
│       ├── label/
│       │   ├── 00000_1_mixed_f80_test.npy  (Note: Assuming .npy, but original code used .json which is unusual)
│       │   └── ... (corresponding label files)
│       ├── fidelity/
│       │   ├── 00000_1_mixed_f80_test.txt  (Note: Assuming .txt, but original code used .json which is unusual)
│       │   └── ... (corresponding fidelity files)
│       └── log_used_files.txt
└── processed_datasets/
    └── dataset_combined_mixed_YYYYMMDD_HHMMSS_DERIVATIVE/  (e.g., dataset_combined_mixed_20250617_150000_DERIVATIVE, this name is derived from `combined_mixed_YYYYMMDD_HHMMSS_COMBINED`)
        ├── dm/
        │   ├── case7_00000_1_mixed_f80_test.json_CURRENT_RUN_TIMESTAMP_SIMULATION.npz
        │   └── ... (all processed density matrices as compressed NumPy files)
        ├── dataset.jsonl
        ├── dataset_cleaned.jsonl
        ├── dataset_train.jsonl
        ├── dataset_val.jsonl
        ├── dataset_test.jsonl
        ├── dataset_creator_log.txt
        └── output/
            ├── YYYYMMDD_HHMMSS_TRAINING_RUN/  (e.g., 20250617_160000_TRAINING_RUN)
            │   ├── model/
            │   │   ├── model_best.pt
            │   │   ├── model_last.pt
            │   │   ├── model_best_full.pt
            │   │   └── model_last_full.pt
            │   ├── logs/
            │   │   ├── summary/
            │   │   │   ├── epoch_01.json
            │   │   │   └── ... (per-epoch summary JSON files)
            │   │   ├── curves/
            │   │   │   ├── training_log.csv
            │   │   │   ├── train_loss_curve.pdf
            │   │   │   ├── validation_loss_curve.pdf
            │   │   │   ├── mae_curve.pdf
            │   │   │   ├── rmse_curve.pdf
            │   │   │   ├── r2_curve.pdf
            │   │   │   ├── smape_curve.pdf
            │   │   │   ├── spearman_corr_curve.pdf
            │   │   │   ├── max_error_curve.pdf
            │   │   │   └── learning_rate_curve.pdf
            │   │   ├── predictions/
            │   │   │   ├── epoch_01_scatter.pdf
            │   │   │   └── ... (per-epoch scatter plots)
            │   │   └── residuals/
            │   │       ├── epoch_01_residuals.pdf
            │   │       └── ... (per-epoch residual histograms)
            │   ├── scalers/
            │   │   ├── scaler_real.pkl
            │   │   ├── scaler_imag.pkl
            │   │   ├── scaler_classical.pkl
            │   │   ├── scaler_target.pkl
            │   │   ├── scaler_diag_real.pkl
            │   │   ├── scaler_diag_imag.pkl
            │   │   └── scaler_diag_imag.pkl
            │   └── test_evaluation/
            │       ├── test_predictions_comparison.csv
            │       ├── scatter_comparison.pdf
            │       ├── residuals_comparison.pdf
            │       ├── error_vs_true.pdf
            │       └── test_metrics_summary.json
```





## Acknowledgements

This work has been supported by the Business Finland through project SeQuSoS (112/31/2024), and Research Council of Finland (Grants No. 365343).

# Contributors

- Shaswato Sarker  - ([@TheCatWalk](https://github.com/TheCatWalk)), *University of Jyväskylä*
- Majid Haghparast - ([@MajidHaghparast](https://github.com/MajidHaghparast)), *University of Jyväskylä*

## Contact

Majid Haghparast <<majid.m.haghparast@jyu.fi>>



