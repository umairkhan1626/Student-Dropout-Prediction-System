"""
EduGuard - Student Retention Portal
====================================
Module: assigner.py
Description:
    Implements the capacity-constrained counselor assignment engine:
    1. Defines the simulated counselor roster across three core specialties:
       - Academic Advising & Curricular Support
       - Financial Aid & Debt Relief
       - Student Life, Retention & Engagement
    2. Maps individual student top SHAP risk drivers to the appropriate counseling specialty.
    3. Executes the prioritized triage assignment rule:
       - Sorts students by descending Referral Priority.
       - Assigns matching specialty with lowest load and available capacity.
       - Fallback to any counselor with capacity.
       - Flags 'Unassigned - capacity full' when institutional capacity is exhausted.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# SIMULATED COUNSELOR ROSTER (Sample roster for institutional demonstration)
# -----------------------------------------------------------------------------
DEFAULT_COUNSELORS = [
    {
        "id": 1,
        "name": "Dr. Elena Rostova",
        "title": "Senior Academic Recovery Advisor",
        "specialty": "Academic",
        "capacity": 300,
        "email": "e.rostova@eduguard.edu",
        "avatar": "👩‍🏫"
    },
    {
        "id": 2,
        "name": "Prof. Marcus Vance",
        "title": "Faculty Tutoring & Curricular Specialist",
        "specialty": "Academic",
        "capacity": 300,
        "email": "m.vance@eduguard.edu",
        "avatar": "👨‍🏫"
    },
    {
        "id": 3,
        "name": "Sarah Jenkins",
        "title": "Emergency Financial Aid Officer",
        "specialty": "Financial",
        "capacity": 250,
        "email": "s.jenkins@eduguard.edu",
        "avatar": "👩‍💼"
    },
    {
        "id": 4,
        "name": "David Kim",
        "title": "Institutional Debt & Tuition Counselor",
        "specialty": "Financial",
        "capacity": 250,
        "email": "d.kim@eduguard.edu",
        "avatar": "👨‍💼"
    },
    {
        "id": 5,
        "name": "Maria Santos",
        "title": "Student Wellbeing & Retention Coordinator",
        "specialty": "Engagement",
        "capacity": 300,
        "email": "m.santos@eduguard.edu",
        "avatar": "👩‍⚕️"
    },
    {
        "id": 6,
        "name": "Dr. Alex Taylor",
        "title": "Adult Learner & Work-Study Mentor",
        "specialty": "Engagement",
        "capacity": 300,
        "email": "a.taylor@eduguard.edu",
        "avatar": "👨‍⚕️"
    }
]


# -----------------------------------------------------------------------------
# FEATURE-TO-SPECIALTY MAPPING
# -----------------------------------------------------------------------------
def map_feature_to_specialty(feature_name: str) -> str:
    """
    Maps a predictive feature to its corresponding advising specialty.

    Viva Explanation:
    - Academic: Curricular units approved/evaluated/grades, admission scores, qualifications.
    - Financial: Tuition fees up to date, debtor status, scholarship holding, macroeconomic indices.
    - Engagement: Student age, marital status, evening attendance, nationality, course choice.
    """
    f = feature_name.lower()

    if any(k in f for k in ["approved", "grade", "evaluations", "enrolled", "credited", "qualification", "admission"]):
        return "Academic"
    elif any(k in f for k in ["tuition", "debtor", "scholarship", "gdp", "inflation", "unemployment"]):
        return "Financial"
    elif any(k in f for k in ["age", "marital", "attendance", "displaced", "special needs", "international", "nacionality", "course"]):
        return "Engagement"
    else:
        return "Academic"  # Default domain


def get_student_top_shap_driver(student_id: int, shap_matrix: np.ndarray, feature_names: List[str]) -> Tuple[str, float]:
    """
    Extracts the single most influential positive SHAP feature pushing this student
    toward dropout. If no positive SHAP exists, returns ('None', 0.0).
    """
    student_shap = shap_matrix[student_id]
    pos_indices = [i for i, val in enumerate(student_shap) if val > 0.0]

    if not pos_indices:
        return ("No Risk Factors", 0.0)

    top_idx = max(pos_indices, key=lambda i: student_shap[i])
    return (feature_names[top_idx], float(student_shap[top_idx]))


# -----------------------------------------------------------------------------
# AUTO-ASSIGNMENT ENGINE (Viva Core Logic)
# -----------------------------------------------------------------------------
def run_auto_assignment_algorithm(
    students_df: pd.DataFrame,
    shap_matrix: np.ndarray,
    feature_names: List[str],
    counselors: List[dict],
    risk_threshold: float = 0.30
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Smart Capacity-Constrained Assignment Algorithm.

    Rule Structure (for Viva Defense):
    1. Filter students with Risk Score >= risk_threshold (High and Medium risk).
    2. Sort students by Referral Priority in descending order (highest triage priority first).
    3. For each student:
       a. Identify top positive SHAP risk factor.
       b. Map factor to specialty ('Academic', 'Financial', 'Engagement').
       c. Find counselors matching the specialty who have free capacity (load < capacity).
          Assign the matching counselor with the lowest current load (load balancing).
       d. If matching counselors are at capacity, assign ANY counselor with free capacity (lowest load).
       e. If all institutional capacity is exhausted, mark as 'Unassigned - capacity full'.
    """
    # 1. Filter High and Medium risk students
    eligible = students_df[students_df["risk_score"] >= risk_threshold].copy()

    # 2. Sort by descending referral priority
    eligible = eligible.sort_values(by="referral_priority", ascending=False)

    # Initialize caseload tracking
    current_loads = {c["name"]: 0 for c in counselors}
    capacity_map = {c["name"]: c["capacity"] for c in counselors}
    specialty_map = {c["name"]: c["specialty"] for c in counselors}

    assignments = {}
    assigned_specialties = {}
    top_drivers = {}

    # 3. Process students in prioritized triage order
    for _, row in eligible.iterrows():
        sid = int(row["student_id"])

        # Identify top SHAP driver
        top_feat, top_shap = get_student_top_shap_driver(sid, shap_matrix, feature_names)
        target_specialty = map_feature_to_specialty(top_feat)
        top_drivers[sid] = top_feat
        assigned_specialties[sid] = target_specialty

        assigned_counselor = None

        # Step 3c: Check matching specialty counselors with available capacity
        matching_counselors = [
            c["name"] for c in counselors
            if c["specialty"] == target_specialty and current_loads[c["name"]] < capacity_map[c["name"]]
        ]

        if matching_counselors:
            # Pick the matching counselor with the lowest load (load balancing)
            assigned_counselor = min(matching_counselors, key=lambda name: current_loads[name])
        else:
            # Step 3d: Fallback to any counselor with available capacity
            available_any = [
                c["name"] for c in counselors
                if current_loads[c["name"]] < capacity_map[c["name"]]
            ]
            if available_any:
                assigned_counselor = min(available_any, key=lambda name: current_loads[name])
            else:
                # Step 3e: Institutional capacity full
                assigned_counselor = "Unassigned - capacity full"

        # Record assignment and increment caseload
        assignments[sid] = assigned_counselor
        if assigned_counselor in current_loads:
            current_loads[assigned_counselor] += 1

    return assignments, assigned_specialties, top_drivers


def get_counselor_stats(counselors: List[dict], assignments: Dict[int, str]) -> List[dict]:
    """
    Calculates real-time caseloads, capacity utilization, and status for each counselor.
    """
    stats = []
    # Count assignments for each counselor
    assigned_series = pd.Series(list(assignments.values())) if assignments else pd.Series(dtype=str)
    counts = assigned_series.value_counts().to_dict()

    for c in counselors:
        name = c["name"]
        load = counts.get(name, 0)
        cap = c["capacity"]
        util = (load / cap) * 100 if cap > 0 else 0.0

        stats.append({
            "id": c["id"],
            "name": name,
            "title": c["title"],
            "specialty": c["specialty"],
            "capacity": cap,
            "current_load": load,
            "available_slots": max(0, cap - load),
            "utilization_pct": round(util, 1),
            "email": c["email"],
            "avatar": c["avatar"]
        })
    return stats
