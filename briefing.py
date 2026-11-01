"""
Student Dropout Risk Prediction and Counselor Briefing System
=============================================================
Module: briefing.py
Description:
    1. Computes student situational urgency scores based on urgent administrative
       and academic triggers (unpaid tuition, debt, zero approved credits).
    2. Calculates Referral Priority for each student using the formula:
           Referral Priority = Risk Score * (1 + urgency_weight * urgency)
       where urgency_weight is fully configurable.
    3. Ranks flagged students (e.g. Risk Score >= 0.50) by referral priority descending
       and exports the ranked triage list to 'ranked_students.csv'.
    4. For any selected student, identifies their actual top 3 positive SHAP risk factors
       with their real-world values from the dataset.
    5. Enforces a strict guardrail: If a student has no positive (risk-increasing) SHAP factors,
       raises NoRiskFactorsError so that no misleading generic briefing is generated.
    6. Formats clean, structured counselor briefings with actionable intervention advice.

Author: University AI Project
Viva Preparation:
    - Referral priority combines statistical risk (the model's probability of dropout)
      with operational urgency (imminent financial or academic disqualification).
    - Explanations are strictly grounded in individual SHAP attributions and raw feature values,
      satisfying AI explainability and algorithmic fairness standards.
    - The NoRiskFactorsError guardrail prevents "hallucinated" or canned generic advice.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib


class NoRiskFactorsError(Exception):
    """
    Exception raised when attempting to generate a risk briefing for a student
    who possesses no positive SHAP values (i.e. no features pushing risk upward).
    Ensures that generic, ungrounded briefings are never produced.
    """
    pass


def load_artifacts():
    """
    Loads precomputed artifacts from Steps 1 and 2:
    - data/data.csv: Raw student features
    - risk_scores.csv: Out-of-fold predicted dropout risk scores
    - features.pkl: Feature names list
    - shap_values.npy: Precomputed Class 1 (Dropout) SHAP values matrix
    """
    data_path = Path("data/data.csv")
    scores_path = Path("risk_scores.csv")
    features_path = Path("features.pkl")
    shap_path = Path("shap_values.npy")

    for p in [data_path, scores_path, features_path, shap_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required artifact '{p}' not found. Please complete Steps 1 & 2 first.")

    df_raw = pd.read_csv(data_path, sep=";")
    df_raw.columns = df_raw.columns.str.strip()

    risk_df = pd.read_csv(scores_path)
    feature_names = joblib.load(features_path)
    shap_values = np.load(shap_path)

    return df_raw, risk_df, feature_names, shap_values


def compute_urgency_scores(df: pd.DataFrame) -> pd.Series:
    """
    Calculates an operational situational urgency score between 0.0 and 1.0.

    Urgency represents imminent risk of administrative or academic dismissal:
        1. Tuition Fees Late (Tuition fees up to date == 0): Weight = 0.4
           - Non-payment blocks future course enrollment and degree conferral.
        2. Institutional Debtor (Debtor == 1): Weight = 0.3
           - Student has outstanding overdue debt requiring financial aid intervention.
        3. Academic Stagnation (Approved == 0 and Enrolled > 0 in 2nd semester): Weight = 0.3
           - Complete course failure in the most recent semester indicates immediate crisis.

    Returns:
        pd.Series: Urgency scores bounded between 0.0 and 1.0.
    """
    # Financial urgency components
    unpaid_tuition = (df["Tuition fees up to date"] == 0).astype(float)
    is_debtor = (df["Debtor"] == 1).astype(float)

    # Academic stagnation component (enrolled in classes but completed none)
    zero_sem2_approved = (
        (df["Curricular units 2nd sem (approved)"] == 0) &
        (df["Curricular units 2nd sem (enrolled)"] > 0)
    ).astype(float)

    # Weighted urgency score (maximum 1.0)
    urgency = (0.4 * unpaid_tuition) + (0.3 * is_debtor) + (0.3 * zero_sem2_approved)
    return urgency.round(4)


def calculate_referral_priorities(
    risk_scores: pd.Series,
    urgency_scores: pd.Series,
    urgency_weight: float = 0.5
) -> pd.Series:
    """
    Calculates referral priority according to the project formula:
        Referral Priority = Risk Score * (1 + urgency_weight * urgency)

    Parameters:
        risk_scores (pd.Series): Predicted dropout risk probability (0.0 to 1.0).
        urgency_scores (pd.Series): Situational urgency score (0.0 to 1.0).
        urgency_weight (float): Configurable weight controlling how much urgency
                                modulates triage order.
    Returns:
        pd.Series: Computed referral priority scores.
    """
    priorities = risk_scores * (1.0 + (urgency_weight * urgency_scores))
    return priorities.round(4)


def rank_students(
    risk_df: pd.DataFrame,
    urgency_series: pd.Series,
    urgency_weight: float = 0.5,
    risk_threshold: float = 0.50
) -> pd.DataFrame:
    """
    Applies the risk threshold to flag at-risk students, computes their referral
    priority, and sorts them in descending priority order for counselor triage.
    """
    ranked = risk_df.copy()
    ranked["urgency"] = urgency_series
    ranked["referral_priority"] = calculate_referral_priorities(
        ranked["risk_score"],
        ranked["urgency"],
        urgency_weight=urgency_weight
    )

    # Flag students whose risk score meets or exceeds the threshold
    flagged = ranked[ranked["risk_score"] >= risk_threshold].copy()
    flagged = flagged.sort_values(by="referral_priority", ascending=False).reset_index(drop=True)
    return flagged


def extract_top_risk_factors(
    student_id: int,
    shap_matrix: np.ndarray,
    feature_names: list,
    df_raw: pd.DataFrame,
    top_n: int = 3
) -> list:
    """
    Extracts the top N positive SHAP risk factors for a specific student,
    along with the student's actual real-world values.

    Guardrail:
        If the student has NO positive SHAP values (i.e. every feature reduces
        dropout probability), raises NoRiskFactorsError. No generic briefing is produced.
    """
    student_shap = shap_matrix[student_id]

    # Filter for strictly positive SHAP values (factors increasing dropout probability)
    positive_indices = [i for i, val in enumerate(student_shap) if val > 0.0]

    if not positive_indices:
        raise NoRiskFactorsError(
            f"Student ID {student_id} has no risk-increasing factors (all SHAP values <= 0). "
            f"A generic briefing cannot be generated."
        )

    # Sort positive indices by SHAP contribution in descending order
    sorted_pos_indices = sorted(positive_indices, key=lambda idx: student_shap[idx], reverse=True)
    selected_indices = sorted_pos_indices[:top_n]

    factors = []
    for idx in selected_indices:
        fname = feature_names[idx]
        real_val = df_raw.loc[student_id, fname]
        shap_val = float(student_shap[idx])
        factors.append({
            "feature": fname,
            "real_value": real_val,
            "shap_value": round(shap_val, 4)
        })

    return factors


def generate_counselor_briefing(
    student_id: int,
    risk_score: float,
    urgency: float,
    referral_priority: float,
    factors: list
) -> str:
    """
    Generates a structured, counselor-ready briefing containing the student's
    actual top 3 risk factors, real feature values, and tailored intervention guidance.
    """
    lines = []
    lines.append("=" * 72)
    lines.append(f"               STUDENT COUNSELOR ADVISING BRIEFING")
    lines.append("=" * 72)
    lines.append(f"Student ID          : {student_id}")
    lines.append(f"Predicted Risk Score: {risk_score * 100:.1f}% (Probability of Dropout)")
    lines.append(f"Situational Urgency : {urgency:.2f} (0.00 = low, 1.00 = critical)")
    lines.append(f"Referral Priority   : {referral_priority:.4f} [Triage Rank Metric]")
    lines.append("-" * 72)
    lines.append("TOP IDENTIFIED RISK DRIVERS (SHAP Local Feature Attributions):")

    action_recommendations = []

    for rank, f in enumerate(factors, start=1):
        feat = f["feature"]
        val = f["real_value"]
        shap_contrib = f["shap_value"]
        lines.append(f"  {rank}. {feat}")
        lines.append(f"     -> Actual Student Value: {val}")
        lines.append(f"     -> Risk Impact (SHAP)  : +{shap_contrib:.4f} toward dropout")

        # Map key features to actionable counseling guidance
        feat_lower = feat.lower()
        if "approved" in feat_lower or "grade" in feat_lower or "evaluations" in feat_lower:
            action_recommendations.append("Academic: Schedule 1-on-1 tutoring review and academic recovery planning.")
        elif "tuition" in feat_lower or "debtor" in feat_lower or "scholarship" in feat_lower:
            action_recommendations.append("Financial: Refer to Student Financial Aid office for emergency payment plan.")
        elif "age" in feat_lower or "marital" in feat_lower or "attendance" in feat_lower:
            action_recommendations.append("Advising: Explore evening tutoring or work-study flexibility for adult learners.")

    lines.append("-" * 72)
    lines.append("RECOMMENDED INTERVENTION FOCUS:")
    unique_actions = list(dict.fromkeys(action_recommendations))
    if unique_actions:
        for act in unique_actions:
            lines.append(f"  * {act}")
    else:
        lines.append("  * Comprehensive holistic review with academic advisor.")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    """
    Main demonstration and verification routine for Step 3.
    """
    print("\n" + "=" * 70)
    print("      STEP 3: COUNSELOR BRIEFING & RANKING SYSTEM")
    print("=" * 70)

    # 1. Load artifacts
    print("1. Loading project artifacts...")
    df_raw, risk_df, feature_names, shap_values = load_artifacts()
    print(f"   -> Loaded {len(df_raw):,} student records and {len(feature_names)} features.")

    # 2. Compute urgency
    print("\n2. Computing situational urgency scores for all students...")
    urgency_scores = compute_urgency_scores(df_raw)
    high_urgency_count = (urgency_scores >= 0.7).sum()
    print(f"   -> Calculated urgency (mean={urgency_scores.mean():.4f}, max={urgency_scores.max():.4f}).")
    print(f"   -> Students with high operational urgency (>= 0.70): {high_urgency_count:,}")

    # 3. Rank flagged students (threshold = 0.50, urgency_weight = 0.5)
    urgency_weight = 0.5
    risk_threshold = 0.50
    print(f"\n3. Ranking flagged students (Risk >= {risk_threshold:.2f}, Urgency Weight = {urgency_weight:.2f})...")
    ranked_flagged = rank_students(
        risk_df,
        urgency_scores,
        urgency_weight=urgency_weight,
        risk_threshold=risk_threshold
    )
    print(f"   -> Flagged {len(ranked_flagged):,} students requiring counseling intervention.")

    # Save to ranked_students.csv
    output_ranked = Path("ranked_students.csv")
    ranked_flagged.to_csv(output_ranked, index=False)
    print(f"   -> Saved triage ranking table to: {output_ranked.resolve()}")

    # Display Top 5 Ranked Students
    print("\n" + "-" * 72)
    print("TOP 5 HIGHEST-PRIORITY REFERRALS (TRIAGE QUEUE)")
    print("-" * 72)
    print(f"{'Rank':<5} | {'Student ID':<11} | {'Risk Score':<11} | {'Urgency':<8} | {'Priority':<10}")
    print("-" * 72)
    for i, row in ranked_flagged.head(5).iterrows():
        print(f"{i + 1:<5} | {int(row['student_id']):<11} | {row['risk_score'] * 100:<10.1f}% | {row['urgency']:<8.2f} | {row['referral_priority']:<10.4f}")
    print("-" * 72)

    # 4. Generate sample briefings for top referrals
    print("\n4. Generating sample counselor briefing for the #1 ranked referral...")
    top_student = ranked_flagged.iloc[0]
    top_id = int(top_student["student_id"])

    top_factors = extract_top_risk_factors(
        top_id,
        shap_values,
        feature_names,
        df_raw,
        top_n=3
    )

    briefing_text = generate_counselor_briefing(
        student_id=top_id,
        risk_score=float(top_student["risk_score"]),
        urgency=float(top_student["urgency"]),
        referral_priority=float(top_student["referral_priority"]),
        factors=top_factors
    )
    print("\n" + briefing_text)

    # 5. Verify the NoRiskFactorsError Guardrail
    print("\n5. Verifying NoRiskFactorsError Guardrail...")
    # Create a synthetic zero-risk student array where all SHAP values are non-positive
    synthetic_zero_risk_shap = np.full((1, len(feature_names)), -0.05)
    try:
        extract_top_risk_factors(
            student_id=0,
            shap_matrix=synthetic_zero_risk_shap,
            feature_names=feature_names,
            df_raw=df_raw,
            top_n=3
        )
        print("   [FAIL] Guardrail did not trigger.")
    except NoRiskFactorsError as e:
        print(f"   [SUCCESS] Guardrail verified! Correctly raised NoRiskFactorsError:")
        print(f"   -> \"{e}\"")

    print("\n[SUCCESS] Step 3 execution completed successfully.\n")


if __name__ == "__main__":
    main()
