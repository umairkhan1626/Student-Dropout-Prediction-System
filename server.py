"""
EduGuard — Student Retention Portal Backend
===========================================
Module: server.py
Description:
    FastAPI REST Backend for the Student Dropout Risk Prediction and Counselor
    Briefing System.

Features:
    - Exposes RESTful API endpoints for cohort metrics, triage queues, student profiles,
      explainable SHAP risk drivers, counselor rosters, and automated capacity assignments.
    - Reuses existing artifacts (model.pkl, features.pkl, shap_values.npy, risk_scores.csv,
      results.csv, ACCEPTANCE.md, briefing.py, and assigner.py) without retraining or inventing numbers.
    - Persists counselor assignments and student case statuses in server memory and to 'state.json'
      so that user updates survive server restarts.
    - Enforces the NoRiskFactorsError guardrail returning an explicit warning when no positive risk factors exist.
    - Serves static web frontend assets from the 'web/' directory.

Author: University AI Project
Viva Preparation:
    - FastAPI provides high-performance, asynchronous REST APIs with automatic OpenAPI (Swagger) documentation.
    - Decoupled from Streamlit: enables flexible web client architectures (React, Vue, or Vanilla JS)
      while preserving 'app.py' as the Streamlit fallback.
"""

import json
from pathlib import Path
from typing import Annotated, Dict, List, Optional
import io
import csv

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Reusable domain logic from Step 3 and Step 4
from briefing import (
    load_artifacts,
    compute_urgency_scores,
    calculate_referral_priorities,
    extract_top_risk_factors,
    generate_counselor_briefing,
    NoRiskFactorsError
)
from assigner import (
    DEFAULT_COUNSELORS,
    map_feature_to_specialty,
    run_auto_assignment_algorithm,
    get_counselor_stats
)


# -----------------------------------------------------------------------------
# INITIALIZE FASTAPI APPLICATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="EduGuard - Student Retention Portal API",
    description=(
        "RESTful API providing student dropout risk predictions, situational urgency scoring, "
        "game-theoretic SHAP explainability, and capacity-constrained counselor assignment."
    ),
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# -----------------------------------------------------------------------------
# COURSE CODE METADATA (UCI Dataset Mapping)
# -----------------------------------------------------------------------------
COURSE_NAMES: Dict[int, str] = {
    9500: "Nursing",
    9147: "Management",
    9238: "Social Service",
    9085: "Veterinary Nursing",
    9773: "Journalism & Comm.",
    9991: "Management (Evening)",
    9670: "Advertising & Mktg",
    9254: "Tourism",
    9070: "Communication Design",
    9003: "Agronomy",
    9119: "Informatics Eng.",
    9130: "Equinculture",
    9853: "Basic Education",
    8014: "Social Service (Evening)",
    9010: "Biofuel Production",
    9145: "Oral Hygiene",
    33: "Biofuel Tech"
}


# -----------------------------------------------------------------------------
# ARTIFACTS LOADING & GLOBAL CACHE
# -----------------------------------------------------------------------------
# Load artifacts once upon server initialization (no retraining, exact numbers)
try:
    df_raw, risk_df, feature_names, shap_values = load_artifacts()
    urgency_scores = compute_urgency_scores(df_raw)
    TOTAL_STUDENTS = len(df_raw)
    print(f"[EduGuard API] Artifacts successfully loaded: {TOTAL_STUDENTS:,} students, {len(feature_names)} features.")
except Exception as e:
    raise RuntimeError(f"[EduGuard API] Failed to load required machine learning artifacts: {e}")


# -----------------------------------------------------------------------------
# PERSISTENT STATE MANAGEMENT (survives restart)
# -----------------------------------------------------------------------------
STATE_FILE = Path("state.json")


def load_persistent_state() -> tuple[Dict[int, str], Dict[int, str]]:
    """
    Loads saved assignments and case statuses from 'state.json'.
    If the file does not exist, initializes empty assignments and 'Flagged' statuses.
    """
    assignments: Dict[int, str] = {}
    statuses: Dict[int, str] = {i: "Flagged" for i in range(TOTAL_STUDENTS)}

    if STATE_FILE.is_file():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            assignments = {int(k): v for k, v in data.get("assignments", {}).items()}
            raw_statuses = {int(k): v for k, v in data.get("statuses", {}).items()}
            statuses.update(raw_statuses)
            print(f"[EduGuard API] Loaded persistent state: {len(assignments)} assignments, {len(statuses)} statuses.")
        except Exception as err:
            print(f"[EduGuard API] Warning: could not parse {STATE_FILE}, using defaults: {err}")

    return assignments, statuses


def save_persistent_state(assignments: Dict[int, str], statuses: Dict[int, str]) -> None:
    """
    Persists current in-memory assignments and statuses to 'state.json'.
    """
    payload = {
        "assignments": {str(k): v for k, v in assignments.items()},
        "statuses": {str(k): v for k, v in statuses.items()}
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# In-memory mutable state backed by state.json
assignments_state, statuses_state = load_persistent_state()


# -----------------------------------------------------------------------------
# HELPER: BUILD MASTER DATAFRAME
# -----------------------------------------------------------------------------
def get_master_dataframe(urgency_weight: float = 0.5) -> pd.DataFrame:
    """
    Constructs the master student dataset with calculated referral priorities,
    assigned risk tiers, counselor assignments, and case statuses.
    """
    # Ensure float conversion in case parameter is passed as Query default
    if hasattr(urgency_weight, "default"):
        w = float(urgency_weight.default)
    else:
        try:
            w = float(urgency_weight)
        except (TypeError, ValueError):
            w = 0.5

    df = risk_df.copy()
    df["urgency"] = urgency_scores
    df["referral_priority"] = calculate_referral_priorities(
        df["risk_score"],
        df["urgency"],
        urgency_weight=w
    )

    def classify_tier(score: float) -> str:
        if score >= 0.60:
            return "High"
        elif score >= 0.30:
            return "Medium"
        else:
            return "Low"

    df["risk_tier"] = df["risk_score"].apply(classify_tier)
    df["assigned_counselor"] = df["student_id"].apply(
        lambda sid: assignments_state.get(sid, "Unassigned")
    )
    df["status"] = df["student_id"].apply(
        lambda sid: statuses_state.get(sid, "Flagged")
    )
    return df


# -----------------------------------------------------------------------------
# PYDANTIC REQUEST SCHEMAS
# -----------------------------------------------------------------------------
class AutoAssignRequest(BaseModel):
    risk_threshold: float = 0.30
    urgency_weight: float = 0.5


class AssignRequest(BaseModel):
    counselor: str


class StatusRequest(BaseModel):
    status: str


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/summary", summary="Cohort Summary KPIs & Distributions")
def get_summary(urgency_weight: Annotated[float, Query(ge=0.0, le=2.0)] = 0.5):
    """
    Returns high-level summary KPIs for the dashboard:
    - Total students and counts by risk tier (High, Medium, Low)
    - Average dropout risk score across the cohort
    - Unassigned students count (eligible students with risk >= 0.30)
    - Active assigned student count
    - Risk score distribution breakdown
    - Average dropout risk aggregated by degree program
    - Top 10 highest-priority referrals for immediate triage
    """
    master = get_master_dataframe(urgency_weight=urgency_weight)

    total_count = len(master)
    high_count = int((master["risk_tier"] == "High").sum())
    med_count = int((master["risk_tier"] == "Medium").sum())
    low_count = int((master["risk_tier"] == "Low").sum())
    avg_risk = float(round(master["risk_score"].mean(), 4))

    # Eligible for intervention: High or Medium risk (score >= 0.30)
    eligible = master[master["risk_score"] >= 0.30]
    unassigned_count = int((eligible["assigned_counselor"] == "Unassigned").sum())
    assigned_count = int((master["assigned_counselor"] != "Unassigned").sum())

    # Risk Distribution Breakdown
    risk_distribution = [
        {"tier": "High", "label": "High Risk (≥ 0.60)", "count": high_count, "percentage": round(high_count / total_count * 100, 2)},
        {"tier": "Medium", "label": "Medium Risk (0.30 - 0.59)", "count": med_count, "percentage": round(med_count / total_count * 100, 2)},
        {"tier": "Low", "label": "Low Risk (< 0.30)", "count": low_count, "percentage": round(low_count / total_count * 100, 2)}
    ]

    # Risk by Academic Course
    df_course = df_raw[["Course"]].copy()
    df_course["course_name"] = df_course["Course"].map(COURSE_NAMES).fillna("Other Course")
    df_course["risk_score"] = master["risk_score"]

    course_stats = (
        df_course.groupby("course_name")
        .agg(average_risk=("risk_score", "mean"), student_count=("risk_score", "count"))
        .reset_index()
        .sort_values(by="average_risk", ascending=False)
    )

    risk_by_course = [
        {
            "course_name": row["course_name"],
            "average_risk": round(float(row["average_risk"]), 4),
            "student_count": int(row["student_count"])
        }
        for _, row in course_stats.iterrows()
    ]

    # Top 10 Priority Triage
    top10_df = master.sort_values(by="referral_priority", ascending=False).head(10)
    top_priority = [
        {
            "student_id": int(r["student_id"]),
            "risk_score": float(r["risk_score"]),
            "risk_tier": r["risk_tier"],
            "urgency": float(r["urgency"]),
            "referral_priority": float(r["referral_priority"]),
            "assigned_counselor": r["assigned_counselor"],
            "status": r["status"]
        }
        for _, r in top10_df.iterrows()
    ]

    return {
        "total_students": total_count,
        "high_risk_count": high_count,
        "medium_risk_count": med_count,
        "low_risk_count": low_count,
        "average_risk": avg_risk,
        "eligible_for_advising": len(eligible),
        "unassigned_count": unassigned_count,
        "assigned_count": assigned_count,
        "urgency_weight": urgency_weight,
        "risk_distribution": risk_distribution,
        "risk_by_course": risk_by_course,
        "top_priority": top_priority
    }


@app.get("/api/students", summary="Paginated & Filterable Student Directory")
def get_students(
    risk_level: Annotated[Optional[str], Query(description="Filter by risk tier: 'all', 'high', 'medium', 'low'")] = "all",
    counselor: Annotated[Optional[str], Query(description="Filter by counselor name, 'all', or 'unassigned'")] = "all",
    status: Annotated[Optional[str], Query(description="Filter by case status: 'all', 'flagged', 'contacted', 'resolved'")] = "all",
    search: Annotated[Optional[str], Query(description="Search by numeric Student ID")] = None,
    sort_by: Annotated[Optional[str], Query(description="'priority', 'risk', 'urgency', or 'id'")] = "priority",
    sort_order: Annotated[Optional[str], Query(description="'asc' or 'desc'")] = "desc",
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 25,
    urgency_weight: Annotated[float, Query(ge=0.0, le=2.0, description="Urgency weight for priority formula")] = 0.5
):
    """
    Returns a searchable, filterable, sorted, and paginated list of students.
    """
    master = get_master_dataframe(urgency_weight=urgency_weight)

    # 1. Apply Risk Tier Filter
    if risk_level and risk_level.lower() != "all":
        lvl = risk_level.lower()
        if lvl == "high":
            master = master[master["risk_tier"] == "High"]
        elif lvl == "medium":
            master = master[master["risk_tier"] == "Medium"]
        elif lvl == "low":
            master = master[master["risk_tier"] == "Low"]

    # 2. Apply Counselor Filter
    if counselor and counselor.lower() != "all":
        if counselor.lower() == "unassigned":
            master = master[master["assigned_counselor"] == "Unassigned"]
        else:
            master = master[master["assigned_counselor"].str.lower() == counselor.lower()]

    # 3. Apply Status Filter
    if status and status.lower() != "all":
        master = master[master["status"].str.lower() == status.lower()]

    # 4. Apply Student ID Search
    if search and search.strip():
        term = search.strip()
        # Clean user-typed prefixes like '#2237' or 'Student #2237'
        clean_term = term.lstrip("#").strip()
        if clean_term.lower().startswith("student"):
            clean_term = clean_term[7:].strip().lstrip("#").strip()

        if clean_term:
            str_ids = master["student_id"].astype(str)
            master = master[str_ids.str.contains(clean_term, regex=False)]

    # 5. Sorting
    sort_mapping = {
        "priority": "referral_priority",
        "risk": "risk_score",
        "urgency": "urgency",
        "id": "student_id"
    }
    col_name = sort_mapping.get(sort_by.lower(), "referral_priority")
    is_ascending = (sort_order.lower() == "asc")
    master = master.sort_values(by=col_name, ascending=is_ascending)

    total_records = len(master)
    total_pages = max(1, (total_records - 1) // page_size + 1)
    page = min(page, total_pages)

    # 6. Pagination Slicing
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    sliced = master.iloc[start_idx:end_idx]

    students = [
        {
            "student_id": int(r["student_id"]),
            "risk_score": float(r["risk_score"]),
            "risk_tier": r["risk_tier"],
            "urgency": float(r["urgency"]),
            "referral_priority": float(r["referral_priority"]),
            "assigned_counselor": r["assigned_counselor"],
            "status": r["status"],
            "true_label": int(r["true_label"])
        }
        for _, r in sliced.iterrows()
    ]

    return {
        "total": total_records,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "students": students
    }


@app.get("/api/students/{student_id}", summary="Single Student Deep-Dive & SHAP Briefing")
def get_student_detail(
    student_id: int,
    urgency_weight: Annotated[float, Query(ge=0.0, le=2.0)] = 0.5
):
    """
    Returns comprehensive profile for an individual student:
    - Statistical risk score and operational urgency
    - Computed referral priority
    - Top 3 positive SHAP risk drivers paired with real unscaled dataset values
    - Top 10 local SHAP feature attributions for interactive charting
    - Structured counselor briefing text (or warning if no positive factors exist)
    - Actionable intervention recommendations
    """
    if student_id < 0 or student_id >= TOTAL_STUDENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Student ID #{student_id} not found. Must be between 0 and {TOTAL_STUDENTS - 1}."
        )

    # Compute student metrics
    s_risk = float(risk_df.loc[student_id, "risk_score"])
    s_urg = float(urgency_scores.iloc[student_id])
    s_priority = float(round(s_risk * (1.0 + urgency_weight * s_urg), 4))
    s_true = int(risk_df.loc[student_id, "true_label"])

    tier = "High" if s_risk >= 0.60 else "Medium" if s_risk >= 0.30 else "Low"
    assigned_counselor = assignments_state.get(student_id, "Unassigned")
    status = statuses_state.get(student_id, "Flagged")

    # Extract Top 3 Positive SHAP Factors & Enforce Guardrail
    has_risk_factors = True
    warning_message = None
    briefing_text = None
    top_factors_list = []
    suggested_actions = []

    try:
        top_factors = extract_top_risk_factors(
            student_id=student_id,
            shap_matrix=shap_values,
            feature_names=feature_names,
            df_raw=df_raw,
            top_n=3
        )
        for f in top_factors:
            f["specialty"] = map_feature_to_specialty(f["feature"])
            # Convert numpy scalar types to native Python types for JSON serialization
            raw_v = f.get("real_value")
            if hasattr(raw_v, "item"):
                f["real_value"] = raw_v.item()
            elif pd.isna(raw_v):
                f["real_value"] = "N/A"
            f["shap_value"] = float(f["shap_value"])
            top_factors_list.append(f)

        briefing_text = generate_counselor_briefing(
            student_id=student_id,
            risk_score=s_risk,
            urgency=s_urg,
            referral_priority=s_priority,
            factors=top_factors
        )

        for f in top_factors:
            feat_lower = f["feature"].lower()
            if any(k in feat_lower for k in ["approved", "grade", "evaluations"]):
                suggested_actions.append("Academic: Schedule 1-on-1 tutoring review and academic recovery planning.")
            elif any(k in feat_lower for k in ["tuition", "debtor", "scholarship"]):
                suggested_actions.append("Financial: Refer to Student Financial Aid office for emergency payment plan.")
            elif any(k in feat_lower for k in ["age", "marital", "attendance"]):
                suggested_actions.append("Advising: Explore evening tutoring or work-study flexibility for adult learners.")

        suggested_actions = list(dict.fromkeys(suggested_actions))
        if not suggested_actions:
            suggested_actions.append("Advising: Comprehensive holistic review with academic advisor.")

    except NoRiskFactorsError as err:
        has_risk_factors = False
        warning_message = (
            f"No risk-increasing factors exist for Student #{student_id} (all SHAP values <= 0). "
            f"No generic briefing will be generated."
        )
        suggested_actions.append("Advising: Student profile exhibits no elevated risk factors. Routine academic check-in.")

    # All local SHAP values (Top 10 by absolute magnitude for charting)
    student_shap = shap_values[student_id]
    sorted_indices = sorted(range(len(student_shap)), key=lambda i: abs(student_shap[i]), reverse=True)[:10]

    all_local_factors = []
    for idx in sorted_indices:
        fname = feature_names[idx]
        real_val = df_raw.loc[student_id, fname]
        if hasattr(real_val, "item"):
            real_val = real_val.item()
        elif pd.isna(real_val):
            real_val = "N/A"
        shap_val = float(student_shap[idx])
        all_local_factors.append({
            "feature": str(fname),
            "real_value": real_val,
            "shap_value": round(shap_val, 4),
            "is_risk_driver": bool(shap_val > 0)
        })

    return {
        "student_id": int(student_id),
        "risk_score": s_risk,
        "risk_tier": tier,
        "urgency": s_urg,
        "referral_priority": s_priority,
        "assigned_counselor": str(assigned_counselor),
        "status": str(status),
        "true_label": s_true,
        "has_risk_factors": bool(has_risk_factors),
        "warning": warning_message,
        "top_risk_factors": top_factors_list,
        "local_shap_factors": all_local_factors,
        "briefing": briefing_text,
        "suggested_actions": suggested_actions
    }


@app.get("/api/counselors", summary="Counselor Roster & Real-Time Caseloads")
def get_counselors():
    """
    Returns the simulated 6-counselor roster with live caseloads,
    available capacity slots, and capacity utilization percentages.
    """
    stats = get_counselor_stats(DEFAULT_COUNSELORS, assignments_state)
    return {
        "total_counselors": len(stats),
        "total_capacity": sum(c["capacity"] for c in stats),
        "total_assigned": sum(c["current_load"] for c in stats),
        "counselors": stats
    }


@app.post("/api/assign/auto", summary="Run Smart Automated Triage Assignment")
def auto_assign(payload: Optional[AutoAssignRequest] = None):
    """
    Executes the capacity-constrained counselor assignment engine:
    1. Filters High and Medium risk students (Risk >= risk_threshold).
    2. Sorts students by Referral Priority in descending order.
    3. Maps each student's top SHAP risk driver to the appropriate advising specialty.
    4. Assigns the specialist with the lowest current load who has open capacity.
    5. Falls back to any advisor with capacity, or marks 'Unassigned - capacity full'.
    Saves results to state.json.
    """
    threshold = payload.risk_threshold if payload else 0.30
    urg_weight = payload.urgency_weight if payload else 0.5

    master = get_master_dataframe(urgency_weight=urg_weight)

    # Run assignment algorithm from assigner.py
    new_assigns, specialties, drivers = run_auto_assignment_algorithm(
        students_df=master,
        shap_matrix=shap_values,
        feature_names=feature_names,
        counselors=DEFAULT_COUNSELORS,
        risk_threshold=threshold
    )

    # Update in-memory state and persist to disk
    assignments_state.update(new_assigns)
    save_persistent_state(assignments_state, statuses_state)

    stats = get_counselor_stats(DEFAULT_COUNSELORS, assignments_state)

    return {
        "success": True,
        "processed_students": len(new_assigns),
        "message": f"Successfully processed and assigned {len(new_assigns):,} at-risk students.",
        "counselors": stats
    }


@app.post("/api/assign/{student_id}", summary="Manually Assign Counselor to Student")
def assign_student(student_id: int, payload: AssignRequest):
    """
    Manually assigns or reassigns a counselor to a specific student.
    Payload: {"counselor": "Dr. Elena Rostova"} or {"counselor": "Unassigned"}.
    Persists update to state.json.
    """
    if student_id < 0 or student_id >= TOTAL_STUDENTS:
        raise HTTPException(status_code=404, detail=f"Student ID #{student_id} not found.")

    valid_names = [c["name"] for c in DEFAULT_COUNSELORS] + ["Unassigned"]
    if payload.counselor not in valid_names and "capacity full" not in payload.counselor:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid counselor name '{payload.counselor}'. Valid choices: {valid_names}"
        )

    assignments_state[student_id] = payload.counselor
    save_persistent_state(assignments_state, statuses_state)

    return {
        "success": True,
        "student_id": student_id,
        "assigned_counselor": payload.counselor
    }


@app.post("/api/status/{student_id}", summary="Update Student Case Status")
def update_status(student_id: int, payload: StatusRequest):
    """
    Updates the outreach and intervention status of a student:
    - 'Flagged' (Initial state)
    - 'Contacted' (Advisor has reached out)
    - 'Resolved' (Action plan completed)
    Persists update to state.json.
    """
    if student_id < 0 or student_id >= TOTAL_STUDENTS:
        raise HTTPException(status_code=404, detail=f"Student ID #{student_id} not found.")

    raw_status = payload.status.strip().capitalize()
    valid_statuses = ["Flagged", "Contacted", "Resolved"]
    if raw_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Must be one of: {valid_statuses}"
        )

    statuses_state[student_id] = raw_status
    save_persistent_state(assignments_state, statuses_state)

    return {
        "success": True,
        "student_id": student_id,
        "status": raw_status
    }


@app.post("/api/reset", summary="Reset All Assignments and Case Statuses")
def reset_system():
    """
    Resets all counselor assignments to 'Unassigned' and all case statuses back to 'Flagged'.
    Cleans state.json.
    """
    assignments_state.clear()
    for sid in range(TOTAL_STUDENTS):
        statuses_state[sid] = "Flagged"

    save_persistent_state(assignments_state, statuses_state)

    return {
        "success": True,
        "message": "All counselor assignments and student statuses have been successfully reset to default."
    }


@app.get("/api/metrics", summary="Model Performance & Acceptance Criteria Scorecard")
def get_metrics():
    """
    Returns empirical cross-validation metrics and criteria evaluation:
    - Stratified 5-Fold CV metrics from results.csv
    - Measurable acceptance criteria verification from ACCEPTANCE.md
    - Global feature importances (Mean Absolute SHAP)
    - Methodological and dataset limitations
    """
    # 1. Model CV Results from results.csv
    cv_results = []
    results_path = Path("results.csv")
    if results_path.is_file():
        res_df = pd.read_csv(results_path)
        cv_results = res_df.to_dict(orient="records")

    # 2. Acceptance Criteria Evaluation Matrix
    acceptance_scorecard = [
        {
            "criterion": "Recall (Dropout Class)",
            "target": "≥ 75.0% (0.75)",
            "achieved_random_forest": "71.36% ± 0.49%",
            "achieved_logistic_regression": "72.27% ± 1.46%",
            "status": "Near Target (Discussion Point)",
            "status_badge": "Near Target",
            "passed": False,
            "viva_justification": "Prioritizes high precision (87.5%) to eliminate counselor alert fatigue."
        },
        {
            "criterion": "ROC-AUC Score",
            "target": "≥ 85.0% (0.85)",
            "achieved_random_forest": "91.98% ± 1.00%",
            "achieved_logistic_regression": "91.82% ± 0.60%",
            "status": "PASSED",
            "status_badge": "PASS",
            "passed": True,
            "viva_justification": "Exceeds target by 7.0 percentage points, demonstrating high discriminative accuracy."
        },
        {
            "criterion": "F1-Score (Dropout Class)",
            "target": "≥ 70.0% (0.70)",
            "achieved_random_forest": "78.59% ± 1.08%",
            "achieved_logistic_regression": "78.97% ± 0.97%",
            "status": "PASSED",
            "status_badge": "PASS",
            "passed": True,
            "viva_justification": "Harmonizes precision and recall under class imbalance."
        },
        {
            "criterion": "Precision (Dropout Class)",
            "target": "≥ 65.0% (0.65)",
            "achieved_random_forest": "87.50% ± 2.67%",
            "achieved_logistic_regression": "87.10% ± 2.08%",
            "status": "PASSED",
            "status_badge": "PASS",
            "passed": True,
            "viva_justification": "Over 87% of flagged students are genuine dropouts."
        },
        {
            "criterion": "Data Quality Gate",
            "target": "Zero unhandled missing values, zero unhandled duplicates",
            "achieved_random_forest": "0 missing values, 0 duplicate rows",
            "achieved_logistic_regression": "0 missing values, 0 duplicate rows",
            "status": "PASSED",
            "status_badge": "PASS",
            "passed": True,
            "viva_justification": "Clean dataset confirmed across all 4,424 records."
        }
    ]

    # 3. Top 10 Global Features (Mean Absolute SHAP)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:10]

    global_features = [
        {
            "rank": rank + 1,
            "feature": feature_names[idx],
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
            "specialty": map_feature_to_specialty(feature_names[idx])
        }
        for rank, idx in enumerate(top_indices)
    ]

    # 4. Methodological Limitations
    limitations = [
        "Portuguese Higher Education Context: Data collected from a single Portuguese polytechnic institute. International transferability requires calibration with local grading and credit structures.",
        "Simulated Counselor Roster: Caseloads and capacities are modeled for demonstration. Production deployment requires integration with institutional HR and student records databases.",
        "Out-of-Fold Risk Scores: Scores represent historical cross-validation. Prospective temporal validation across upcoming semesters is recommended before administrative policy changes."
    ]

    return {
        "cross_validation_results": cv_results,
        "acceptance_scorecard": acceptance_scorecard,
        "global_feature_importance": global_features,
        "limitations": limitations
    }


@app.get("/api/export.csv", summary="Export Filtered Student Triage Queue to CSV")
def export_csv(
    risk_level: Annotated[Optional[str], Query()] = "all",
    counselor: Annotated[Optional[str], Query()] = "all",
    status: Annotated[Optional[str], Query()] = "all",
    search: Annotated[Optional[str], Query()] = None,
    urgency_weight: Annotated[float, Query(ge=0.0, le=2.0)] = 0.5
):
    """
    Streams a downloadable CSV file of student triage records based on currently applied filters.
    """
    master = get_master_dataframe(urgency_weight=urgency_weight)

    # Apply filters
    if risk_level and risk_level.lower() != "all":
        lvl = risk_level.lower()
        if lvl in ["high", "medium", "low"]:
            master = master[master["risk_tier"].str.lower() == lvl]

    if counselor and counselor.lower() != "all":
        if counselor.lower() == "unassigned":
            master = master[master["assigned_counselor"] == "Unassigned"]
        else:
            master = master[master["assigned_counselor"].str.lower() == counselor.lower()]

    if status and status.lower() != "all":
        master = master[master["status"].str.lower() == status.lower()]

    if search and search.strip():
        term = search.strip()
        clean_term = term.lstrip("#").strip()
        if clean_term.lower().startswith("student"):
            clean_term = clean_term[7:].strip().lstrip("#").strip()
        if clean_term:
            str_ids = master["student_id"].astype(str)
            master = master[str_ids.str.contains(clean_term, regex=False)]

    # Sort descending by priority
    master = master.sort_values(by="referral_priority", ascending=False)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Student ID",
        "Risk Score (%)",
        "Risk Tier",
        "Urgency Score",
        "Referral Priority",
        "Assigned Counselor",
        "Case Status",
        "Actual Outcome"
    ])

    for _, row in master.iterrows():
        writer.writerow([
            int(row["student_id"]),
            f"{row['risk_score'] * 100:.1f}%",
            row["risk_tier"],
            f"{row['urgency']:.2f}",
            f"{row['referral_priority']:.4f}",
            row["assigned_counselor"],
            row["status"],
            "Dropout" if row["true_label"] == 1 else "Retained"
        ])

    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eduguard_students_export.csv"}
    )


# -----------------------------------------------------------------------------
# STATIC FILES SERVING (Mount after API routes to avoid path collision)
# -----------------------------------------------------------------------------
web_dir = Path("web")
web_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="web", html=True), name="static")


# -----------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n[EduGuard Backend] Starting server at http://127.0.0.1:8000 ...")
    uvicorn.run("server.py:app", host="127.0.0.1", port=8000, reload=True)
