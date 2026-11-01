"""
Student Dropout Risk Prediction and Counselor Briefing System
=============================================================
Module: train.py
Description:
    1. Loads student academic data from 'data/data.csv' (semicolon-separated).
    2. Runs comprehensive data quality checks (shape, missing values, duplicates, class balance).
    3. Formulates a binary target: Dropout = 1 (at-risk), Graduate/Enrolled = 0.
    4. Evaluates and compares Logistic Regression (Baseline) and Random Forest Classifier
       using Stratified 5-Fold Cross-Validation.
    5. Reports Precision, Recall, F1-Score, and ROC-AUC (mean and standard deviation).
    6. Saves cross-validation comparison results to 'results.csv'.

Author: University AI Project
Viva Preparation:
    - Stratified K-Fold is used to preserve the exact dropout ratio across all folds.
    - Pipelines are used with StandardScaler for Logistic Regression to prevent data leakage.
    - Recall is prioritized because failing to detect an at-risk student (False Negative)
      is far more damaging than offering extra advising to a stable student (False Positive).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def check_dataset_exists(file_path: Path) -> None:
    """
    Verifies that the dataset file exists before proceeding.
    If missing, provides a helpful instruction message and exits.
    """
    if not file_path.is_file():
        print(f"\n[ERROR] Dataset not found at: {file_path.resolve()}")
        print("Please place the UCI / Kaggle 'Predict Students Dropout and Academic Success'")
        print(f"dataset file at '{file_path}' (semicolon-separated ';') and run this script again.\n")
        sys.exit(1)


def run_data_quality_checks(df: pd.DataFrame, target_col: str) -> None:
    """
    Performs and prints data quality assessments:
    - Dataset dimensions (rows, columns)
    - Missing values count & percentage
    - Duplicate rows count
    - Original class balance for the Target column
    """
    print("\n" + "=" * 60)
    print("           DATA QUALITY & INTEGRITY CHECKS")
    print("=" * 60)

    # 1. Dimensions
    num_rows, num_cols = df.shape
    print(f"1. Dataset Dimensions: {num_rows:,} rows, {num_cols} columns")

    # 2. Missing Values
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"2. Missing Values Check: {total_missing} total missing values across all cells.")
    if total_missing > 0:
        cols_with_missing = missing_series[missing_series > 0]
        print("   Columns with missing values:")
        for col, count in cols_with_missing.items():
            pct = (count / num_rows) * 100
            print(f"     - {col}: {count} ({pct:.2f}%)")
    else:
        print("   -> Clean dataset: No missing values detected.")

    # 3. Duplicate Rows
    duplicates_count = df.duplicated().sum()
    print(f"3. Duplicate Rows Check: {duplicates_count} duplicate rows found.")
    if duplicates_count > 0:
        print(f"   -> Note: {duplicates_count} duplicate rows exist in the raw dataset.")
    else:
        print("   -> Clean dataset: No duplicate records found.")

    # 4. Original Target Class Distribution
    print(f"4. Original '{target_col}' Class Distribution:")
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    for label, count in counts.items():
        print(f"     - {label:12s}: {count:,} students ({percentages[label]:.2f}%)")


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Binarizes the target column and separates features (X) and target (y).
    Target mapping:
        - 'Dropout'           -> 1 (At-Risk student requiring intervention)
        - 'Graduate'/'Enrolled' -> 0 (Remaining in the academic program)
    """
    # Clean whitespace in column names if any
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' was not found in dataset columns: {list(df.columns)}")

    # Binary encoding: Dropout = 1, everything else = 0
    y = (df[target_col].astype(str).str.strip().str.lower() == "dropout").astype(int)
    X = df.drop(columns=[target_col])

    print("\n" + "=" * 60)
    print("          TARGET BINARIZATION & CLASS BALANCE")
    print("=" * 60)
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    total = len(y)
    print(f"Positive Class (Dropout = 1)      : {pos_count:,} ({pos_count / total * 100:.2f}%)")
    print(f"Negative Class (Non-Dropout = 0)  : {neg_count:,} ({neg_count / total * 100:.2f}%)")
    print(f"Total Student Records             : {total:,}")
    print(f"Total Predictive Features         : {X.shape[1]}")
    print("=" * 60)

    return X, y


def build_models():
    """
    Initializes models for comparison:
    1. Baseline: Logistic Regression with StandardScaler.
       - StandardScaler is placed inside a Pipeline so that it fits ONLY on training
         folds during cross-validation, strictly preventing data leakage.
    2. Random Forest:
       - Tree-based ensemble with 100 estimators.
       - Handles non-linear feature interactions naturally.
    """
    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def evaluate_models_cv(models: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """
    Executes Stratified 5-Fold Cross-Validation for each model.
    Evaluates:
        - Precision: Out of all students predicted as dropout, how many actually dropped out?
        - Recall: Out of all actual dropouts, how many did the model correctly identify?
        - F1-Score: Harmonic mean of precision and recall.
        - ROC-AUC: Area under the ROC curve (discriminative ability across thresholds).

    Returns a pandas DataFrame summarizing the mean and standard deviation of each metric.
    """
    print(f"\nEvaluating models using Stratified {n_splits}-Fold Cross-Validation...")

    # Stratified K-Fold ensures every split has the same proportion of dropouts
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results_list = []

    for name, model in models.items():
        print(f"  -> Running cross-validation for: {name}...")
        cv_output = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        prec_mean = float(np.mean(cv_output["test_precision"]))
        prec_std = float(np.std(cv_output["test_precision"]))

        rec_mean = float(np.mean(cv_output["test_recall"]))
        rec_std = float(np.std(cv_output["test_recall"]))

        f1_mean = float(np.mean(cv_output["test_f1"]))
        f1_std = float(np.std(cv_output["test_f1"]))

        roc_mean = float(np.mean(cv_output["test_roc_auc"]))
        roc_std = float(np.std(cv_output["test_roc_auc"]))

        results_list.append({
            "Model": name,
            "Precision_Mean": round(prec_mean, 4),
            "Precision_Std": round(prec_std, 4),
            "Recall_Mean": round(rec_mean, 4),
            "Recall_Std": round(rec_std, 4),
            "F1_Mean": round(f1_mean, 4),
            "F1_Std": round(f1_std, 4),
            "ROC_AUC_Mean": round(roc_mean, 4),
            "ROC_AUC_Std": round(roc_std, 4)
        })

    results_df = pd.DataFrame(results_list)
    return results_df


def print_and_save_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Displays the cross-validation comparison table in a human-readable format
    and writes the results to 'results.csv'.
    """
    print("\n" + "=" * 90)
    print("                    STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 90)
    print(f"{'Model':<32} | {'Precision':<13} | {'Recall':<13} | {'F1-Score':<13} | {'ROC-AUC':<13}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        p_str = f"{row['Precision_Mean']:.4f} ± {row['Precision_Std']:.4f}"
        r_str = f"{row['Recall_Mean']:.4f} ± {row['Recall_Std']:.4f}"
        f_str = f"{row['F1_Mean']:.4f} ± {row['F1_Std']:.4f}"
        auc_str = f"{row['ROC_AUC_Mean']:.4f} ± {row['ROC_AUC_Std']:.4f}"
        print(f"{row['Model']:<32} | {p_str:<13} | {r_str:<13} | {f_str:<13} | {auc_str:<13}")
    print("=" * 90)

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Cross-validation results successfully saved to: {output_path.resolve()}\n")


def main():
    """
    Main execution pipeline for Step 1.
    """
    data_file = Path("data/data.csv")
    results_file = Path("results.csv")
    target_column = "Target"

    # 1. Check if dataset exists
    check_dataset_exists(data_file)

    # 2. Load dataset
    print(f"\nLoading dataset from '{data_file}'...")
    # The UCI dataset is semicolon-separated
    df = pd.read_csv(data_file, sep=";")

    # 3. Data quality checks
    run_data_quality_checks(df, target_col=target_column)

    # 4. Target binarization & feature preparation
    X, y = prepare_features_and_target(df, target_col=target_column)

    # 5. Build models
    models = build_models()

    # 6. Stratified 5-Fold Cross-Validation evaluation
    results_df = evaluate_models_cv(models, X, y, n_splits=5)

    # 7. Print and save results
    print_and_save_results(results_df, results_file)


if __name__ == "__main__":
    main()
