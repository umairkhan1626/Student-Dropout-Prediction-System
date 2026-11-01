"""
Student Dropout Risk Prediction and Counselor Briefing System
=============================================================
Module: explain.py
Description:
    1. Loads student academic data from 'data/data.csv' (semicolon-separated).
    2. Builds the binary target: Dropout = 1 (at-risk), Graduate/Enrolled = 0.
    3. Trains the final Random Forest classifier on the entire dataset
       with class_weight="balanced" and random_state=42.
    4. Computes out-of-fold (OOF) predicted probabilities using Stratified 5-Fold
       cross-validation and saves them to 'risk_scores.csv' (student index, risk score, true label).
    5. Applies SHAP TreeExplainer to the final model, robustly extracting class 1 (Dropout)
       SHAP values across all SHAP library versions.
    6. Generates and saves a global SHAP summary bar plot for top 15 features as 'shap_summary.png'.
    7. Persists artifacts: 'model.pkl' (trained model), 'features.pkl' (feature names),
       and 'shap_values.npy' (class 1 SHAP values matrix).
    8. Prints the top 10 most influential features ranked by mean absolute SHAP value.

Author: University AI Project
Viva Preparation:
    - Out-of-fold (OOF) scoring ensures risk scores for historical students are realistic
      and unbiased (avoiding training set overconfidence).
    - SHAP (SHapley Additive exPlanations) grounds individual predictions in cooperative
      game theory, calculating the exact additive impact of each feature toward dropout risk.
    - Exported artifacts (model, features, shap values, risk scores) decouple expensive training
      from the real-time counselor briefing engine and Streamlit dashboard.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Use non-interactive backend for reliable figure saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import shap


def check_dataset_exists(file_path: Path) -> None:
    """
    Ensures data/data.csv exists before continuing.
    Provides clear feedback if the dataset has not been placed yet.
    """
    if not file_path.is_file():
        print(f"\n[ERROR] Dataset not found at: {file_path.resolve()}")
        print("Please place the UCI / Kaggle 'Predict Students Dropout and Academic Success'")
        print(f"dataset file at '{file_path}' (semicolon-separated ';') and run this script again.\n")
        sys.exit(1)


def load_and_preprocess_data(data_path: Path, target_col: str = "Target"):
    """
    Reads the semicolon-separated CSV and maps target categories:
        - 'Dropout' -> 1 (Positive / At-Risk class)
        - 'Graduate' / 'Enrolled' -> 0 (Negative / Retained class)
    Returns:
        X (pd.DataFrame): feature matrix
        y (pd.Series): binary targets
    """
    print(f"\n1. Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path, sep=";")
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    # Target binarization
    y = (df[target_col].astype(str).str.strip().str.lower() == "dropout").astype(int)
    X = df.drop(columns=[target_col])

    print(f"   -> Loaded {len(df):,} student records with {X.shape[1]} predictive features.")
    print(f"   -> Target distribution: {y.sum():,} Dropouts (Class 1), {(len(y) - y.sum()):,} Retained (Class 0).")
    return X, y


def compute_out_of_fold_risk_scores(X: pd.DataFrame, y: pd.Series, output_path: Path, n_splits: int = 5) -> pd.DataFrame:
    """
    Computes out-of-fold predicted dropout probabilities for all students using
    Stratified 5-Fold cross-validation.

    Why Out-of-Fold (OOF)?
    If we evaluate risk scores on the exact data used to train the tree, the model would
    exhibit overconfidence. OOF probabilities guarantee that each student's score was
    computed by a model trained without seeing that student's record.
    """
    print(f"\n2. Computing out-of-fold risk scores via Stratified {n_splits}-Fold cross-validation...")

    rf_cv = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # method="predict_proba" outputs an array of shape (n_samples, 2): [prob_class_0, prob_class_1]
    oof_probabilities = cross_val_predict(
        rf_cv,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )

    # Class 1 probability represents the predicted Dropout Risk Score (0.0 to 1.0)
    risk_scores = oof_probabilities[:, 1]

    risk_df = pd.DataFrame({
        "student_id": X.index,
        "risk_score": np.round(risk_scores, 4),
        "true_label": y.values
    })

    risk_df.to_csv(output_path, index=False)
    print(f"   -> Out-of-fold risk scores saved to: {output_path.resolve()}")
    return risk_df


def train_final_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Trains the production Random Forest classifier on the entire dataset.
    Settings match train.py:
        - n_estimators = 100
        - class_weight = 'balanced' (counteracts class imbalance)
        - random_state = 42 (ensures reproducibility)
    """
    print("\n3. Training final Random Forest model on the complete dataset...")
    final_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    final_rf.fit(X, y)
    print("   -> Final model training completed successfully.")
    return final_rf


def extract_class1_shap_values(raw_shap_values) -> np.ndarray:
    """
    Extracts the 2D matrix (n_samples, n_features) corresponding specifically
    to Class 1 (Dropout) across different versions of the SHAP library.

    SHAP Version Compatibility:
    - Older SHAP versions return a Python list: [array_class_0, array_class_1].
    - Newer SHAP versions return a 3D numpy array: (n_samples, n_features, n_classes).
    - Explanation objects wrap values under .values.
    """
    vals = raw_shap_values.values if hasattr(raw_shap_values, "values") else raw_shap_values

    if isinstance(vals, list):
        if len(vals) == 2:
            return np.array(vals[1])
        return np.array(vals[0])

    if isinstance(vals, np.ndarray):
        if vals.ndim == 3 and vals.shape[2] == 2:
            return vals[:, :, 1]
        if vals.ndim == 2:
            return vals

    raise ValueError(f"Unrecognized SHAP output format with type: {type(raw_shap_values)}")


def compute_shap_explanations(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Constructs a SHAP TreeExplainer and computes Shapley values for all students.
    Returns:
        np.ndarray: 2D array of shape (n_students, n_features) for the Dropout class.
    """
    print("\n4. Applying SHAP TreeExplainer to the final Random Forest model...")
    explainer = shap.TreeExplainer(model)
    raw_shap_vals = explainer.shap_values(X)

    # Standardize to 2D numpy array for Class 1 (Dropout)
    shap_class1 = extract_class1_shap_values(raw_shap_vals)
    print(f"   -> Successfully extracted Class 1 (Dropout) SHAP matrix with shape: {shap_class1.shape}")
    return shap_class1


def save_global_summary_plot(shap_class1: np.ndarray, X: pd.DataFrame, output_path: Path) -> None:
    """
    Generates and saves a global SHAP summary bar chart highlighting
    the top 15 features that drive student dropout predictions.
    """
    print(f"\n5. Generating global SHAP summary plot (Top 15 features)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_class1,
        X,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("Top 15 Predictors of Student Dropout Risk (SHAP Global Importance)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> Global summary plot saved to: {output_path.resolve()}")


def save_artifacts(model: RandomForestClassifier, feature_names: list, shap_values: np.ndarray) -> None:
    """
    Saves the final trained model, feature names list, and SHAP matrix.
    These artifacts are later loaded directly by the counselor briefing system
    and Streamlit application for fast inference without recomputation.
    """
    print("\n6. Persisting model and explanation artifacts...")
    model_path = Path("model.pkl")
    features_path = Path("features.pkl")
    shap_path = Path("shap_values.npy")

    joblib.dump(model, model_path)
    print(f"   -> Saved model to: {model_path.resolve()}")

    joblib.dump(feature_names, features_path)
    print(f"   -> Saved feature names list to: {features_path.resolve()}")

    np.save(shap_path, shap_values)
    print(f"   -> Saved Class 1 SHAP matrix to: {shap_path.resolve()}")


def print_top_features_table(shap_class1: np.ndarray, feature_names: list, top_n: int = 10) -> None:
    """
    Computes and displays the top N most important features ranked by
    their mean absolute SHAP value across all students.
    """
    mean_abs_shap = np.mean(np.abs(shap_class1), axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 66)
    print(f"     TOP {top_n} MOST IMPORTANT FEATURES (MEAN ABSOLUTE SHAP)")
    print("=" * 66)
    print(f"{'Rank':<5} | {'Feature Name':<42} | {'Mean |SHAP|':<12}")
    print("-" * 66)
    for rank, row in importance_df.head(top_n).iterrows():
        print(f"{rank + 1:<5} | {row['Feature']:<42} | {row['Mean_Abs_SHAP']:.4f}")
    print("=" * 66)


def main():
    """
    Main orchestration function for Step 2.
    """
    data_path = Path("data/data.csv")
    risk_scores_path = Path("risk_scores.csv")
    summary_plot_path = Path("shap_summary.png")

    # Step A: Validate dataset existence
    check_dataset_exists(data_path)

    # Step B: Load dataset & binarize target
    X, y = load_and_preprocess_data(data_path, target_col="Target")

    # Step C: Compute out-of-fold risk scores via Stratified 5-Fold CV
    compute_out_of_fold_risk_scores(X, y, output_path=risk_scores_path, n_splits=5)

    # Step D: Train final model on entire dataset
    final_rf = train_final_model(X, y)

    # Step E: Compute SHAP explanations for the Dropout class (Class 1)
    shap_class1 = compute_shap_explanations(final_rf, X)

    # Step F: Generate and save global SHAP summary plot (Top 15 features)
    save_global_summary_plot(shap_class1, X, output_path=summary_plot_path)

    # Step G: Save model.pkl, features.pkl, and shap_values.npy
    save_artifacts(final_rf, list(X.columns), shap_class1)

    # Step H: Display Top 10 features table
    print_top_features_table(shap_class1, list(X.columns), top_n=10)

    print("\n[SUCCESS] Step 2 execution completed successfully.\n")


if __name__ == "__main__":
    main()
