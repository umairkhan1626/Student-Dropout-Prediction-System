# Project Acceptance Criteria & Assumptions

**Project Title:** Student Dropout Risk Prediction and Counselor Briefing System  
**Author:** Individual University AI Project  
**Target Domain:** Higher Education Academic Advising & Early Intervention  

---

## 1. Problem Overview & Objective
Higher education institutions face significant retention challenges. Early identification of students at risk of academic failure or dropping out allows academic advisors and counselors to deploy targeted interventions (such as tutoring, financial advising, or mental health counseling) before a student officially withdraws.

The objective of this machine learning system is:
1. To accurately identify students at risk of dropping out (Binary Classification: `Dropout = 1`, `Non-Dropout = 0`).
2. To explain *why* a particular student is flagged using interpretable SHAP (SHapley Additive exPlanations) values.
3. To compute a referral priority score combining predicted risk and situational urgency to assist counselors with case triage.

---

## 2. Measurable Acceptance Criteria

To ensure the model is practical and safe for deployment in academic counseling, it must satisfy the following quantitative performance metrics evaluated via **Stratified 5-Fold Cross-Validation**:

| Metric | Target Threshold | Motivation / Justification |
| :--- | :--- | :--- |
| **Recall (Dropout Class)** | **$\ge 0.75$ (75%)** | **Primary metric.** Missing an at-risk student (False Negative) has high cost—the student may drop out without receiving assistance. A recall of at least 75% ensures the majority of vulnerable students are flagged. |
| **ROC-AUC** | **$\ge 0.85$ (85%)** | Measures overall discriminative ability across all possible classification probability thresholds, independent of class imbalance. |
| **F1-Score (Dropout Class)** | **$\ge 0.70$ (70%)** | Harmonizes precision and recall, ensuring that high recall is not achieved simply by trivial over-flagging of all students. |
| **Precision (Dropout Class)** | **$\ge 0.65$ (65%)** | Minimizes counselor alert fatigue. If too many non-dropout students are flagged (False Positives), counseling staff capacity will be exhausted. |
| **Data Quality Gate** | **Zero unhandled missing values, zero unhandled duplicates** | Ensures training and inference integrity on the semicolon-delimited UCI dataset before modeling. |

---

## 3. Key Assumptions

1. **Dataset Integrity & Format**:
   - The dataset used is the UCI / Kaggle *"Predict Students' Dropout and Academic Success"* dataset placed at `data/data.csv`.
   - The file is semicolon-separated (`;`) and contains a `Target` column with three original categories: `Dropout`, `Graduate`, and `Enrolled`.
   - All predictive features are available at the time of assessment (end of semester 1 / semester 2).

2. **Binary Target Formulation**:
   - `Dropout` is mapped to positive class **`1`** (the event of interest).
   - `Graduate` and `Enrolled` are mapped to negative class **`0`** (students remaining in the academic system).

3. **Timeliness & Temporal Availability**:
   - Features represent information collected upon enrollment (demographic, socio-economic, admission marks) and academic progress across the 1st and 2nd semesters (curricular units enrolled, approved, evaluations, grades).
   - Predictions are intended to be run at key advising milestones (e.g., end of 1st year / semester) so interventions remain actionable.

4. **Ethical & Non-Punitive Use**:
   - Predictions and risk scores serve purely as **assistive decision support** for supportive interventions (tutoring, advising, financial aid consultations).
   - Model predictions must never be used for punitive, disciplinary, or exclusionary academic actions.

5. **Explainability Requirement**:
   - Every flagged student briefing must present student-specific top SHAP risk factors with their exact, unscaled real values.
   - If a model flags a student but no positive risk-increasing factor exists, the briefing generator must raise an explicit exception rather than emitting a generic or misleading explanation.
