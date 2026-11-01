# Project Progress & Handoff Report
**Project Name:** EduGuard — Student Dropout Risk Prediction & Counselor Briefing System  
**Last Updated:** November 1, 2026  
**Target Submission Deadline:** November 15, 2026 (UMS Submission)  

---

## 1. What Has Been Completed So Far

### Step 1: Data Integrity, Preprocessing & 5-Fold Cross-Validation
- **`data/data.csv`**: Loaded the official UCI "Predict Students' Dropout and Academic Success" dataset containing 4,424 student records and 36 predictive features (semicolon-delimited).
- **`ACCEPTANCE.md`**: Defined explicit, measurable acceptance criteria:
  - Recall (Dropout Class) $\ge 0.75$
  - ROC-AUC Score $\ge 0.85$
  - F1-Score $\ge 0.70$
  - Precision $\ge 0.65$
  - Data Quality Gate: Zero missing cells, zero duplicates.
- **`requirements.txt`**: Pinned core dependencies (`pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `shap`, `streamlit`).
- **`train.py`**:
  - Validated data quality (shape, zero missing values, zero duplicates, target distribution).
  - Formulated binary target: `Dropout = 1` (1,421 students, 32.12%), `Enrolled / Graduate = 0` (3,003 students, 67.88%).
  - Evaluated Baseline Logistic Regression (using `Pipeline` with `StandardScaler` to prevent data leakage) against Random Forest Classifier across Stratified 5-Fold Cross-Validation.
- **`results.csv`**: Saved empirical cross-validation metrics (mean ± std):
  - *Random Forest:* Precision: **87.50% ± 2.67%**, Recall: **71.36% ± 0.49%**, F1: **78.59% ± 1.08%**, ROC-AUC: **0.9198 ± 0.0100**
  - *Logistic Regression:* Precision: **87.10% ± 2.08%**, Recall: **72.27% ± 1.46%**, F1: **78.97% ± 0.97%**, ROC-AUC: **0.9182 ± 0.0060**

### Step 2: Out-of-Fold Risk Scoring & SHAP Explainability
- **`explain.py`**:
  - Computed honest, unbiased Out-of-Fold (OOF) predicted probabilities using `cross_val_predict` with `StratifiedKFold(n_splits=5)`.
  - Trained the final production Random Forest model on the complete dataset (`class_weight="balanced"`).
  - Applied SHAP `TreeExplainer` across all 4,424 students, with backward/forward compatibility handling for 2D, 3D, and Explanation output formats for Class 1 (Dropout).
  - Generated and saved the global feature importance bar chart (`shap_summary.png`).
- **`risk_scores.csv`**: Out-of-fold predicted dropout risk scores (`student_id`, `risk_score`, `true_label`).
- **`model.pkl`**: Serialized production Random Forest model.
- **`features.pkl`**: Serialized list of the 36 feature column names.
- **`shap_values.npy`**: Precomputed Class 1 SHAP matrix (4,424 rows $\times$ 36 features).
- **`shap_summary.png`**: Top 15 global predictors plot (Top 3: `Curricular units 2nd sem (approved)`, `grade`, and `1st sem (approved)`).

### Step 3: Triage Priority Ranking & Counselor Briefings
- **`briefing.py`**:
  - Implemented situational urgency score ($0.0 \text{ to } 1.0$) based on imminent administrative and academic dismissal factors (unpaid tuition fees, debtor status, and zero approved units in 2nd semester).
  - Implemented Referral Priority formula:
    $$\text{Referral Priority} = \text{Risk Score} \times (1 + \text{urgency\_weight} \times \text{urgency})$$
  - Extracted student-specific top 3 positive SHAP risk drivers paired with actual unscaled values from `data.csv`.
  - Implemented `NoRiskFactorsError` safety guardrail to halt generic or hallucinated briefings if no positive risk factors exist.
- **`ranked_students.csv`**: Exported prioritized triage queue of 1,353 flagged students ($\text{Risk} \ge 0.50$).

### Step 4: Web Application Architecture
- **`styles.py`**: Injected custom CSS for Navy/Slate SaaS theme, white cards, soft shadows, colored risk badges (High, Medium, Low), status badges (Flagged, Contacted, Resolved), and top brand header.
- **`assigner.py`**: Defined simulated 6-advisor roster (Academic, Financial, Engagement specialties with 1,700 total student capacity), feature-to-specialty mapping, and capacity-constrained auto-assignment algorithm.
- **`app.py`**: Multi-page Streamlit portal ("EduGuard - Student Retention Portal") with zero deprecation warnings (`width='stretch'`, no `inplace=True`).
- **`README.md`**: Complete project documentation, architecture diagram, formulas, and execution steps.
- **`VIVA_PREP.md`**: Oral examination defense guide covering 13 common questions with real empirical numbers.

---

## 2. Current State of `app.py`

### Implemented Pages & Features:
1. **Top Brand Banner**: EduGuard logo, subtitle, and dynamic date.
2. **Sidebar Navigation**:
   - Radio buttons with icons for all 7 pages.
   - Urgency weight slider ($0.0 \text{ to } 2.0$, default $0.5$).
   - Risk tier reference guide (High $\ge 0.60$, Medium $0.30 - 0.59$, Low $< 0.30$).
3. **Dashboard Page**:
   - 6 KPI cards (Total Cohort, High Risk, Medium Risk, Low Risk, Average Risk, Unassigned).
   - Risk Level Distribution bar chart (High, Medium, Low counts).
   - Average Dropout Risk by Degree Course horizontal bar chart.
   - Top 10 Priority Triage Queue with individual "View" buttons navigating to Student Profile.
4. **Students Directory Page**:
   - Search by numeric Student ID.
   - Filter by Risk Level (All, High, Medium, Low), Counselor (All, Unassigned, specific), and Status.
   - Pagination controls (15, 30, 50, 100 rows per page; Previous / Next buttons).
   - "View" and "Assign" buttons per row.
   - CSV export of filtered students.
5. **Student Profile Page**:
   - Header summary card with Risk Badge, Status Badge, Risk %, Urgency, and Priority.
   - Horizontal SHAP local feature attribution chart (Red for risk drivers, Blue for protective factors) with real values.
   - Official Advising Briefing text area and `.txt` file download.
   - Action buttons:
     - "🤖 Auto-Assign by Specialty"
     - "🔄 Reassign Counselor" (opens `@st.dialog` pop-up)
     - "📞 Mark as Contacted" (updates state with `st.toast`)
     - "✅ Mark as Resolved" (updates state with `st.toast`)
6. **Counselors Page**:
   - 3x2 grid of simulated counselor cards with specialty badges, live caseload progress bars, capacity, and utilization percentage.
   - "View Caseload" button filtering the directory to that advisor.
7. **Assignments Page**:
   - "⚡ Run Auto-Assignment for High & Medium Students" button executing the prioritized specialty-matching rule.
   - "🔄 Reset All Assignments" button.
   - Real-time advisor utilization summary table.
   - Current assignment records table and CSV download.
8. **Reports Page**:
   - Acceptance Criteria scorecard matrix comparing target thresholds against achieved metrics.
   - 5-Fold cross-validation results table (`results.csv`).
   - Global SHAP feature importance plot (`shap_summary.png`).
   - Methodological & Practical Limitations callout box.
9. **About Page**:
   - Problem statement, 7-step technical pipeline, and technology stack.

### Known Limitations / Current State Notes:
- **No broken syntax or runtime errors**: All scripts compile cleanly.
- **In-Memory State**: Counselor assignments and student contact statuses live in Streamlit's `st.session_state`. Refreshing the browser resets dynamic assignments back to default ("Unassigned"), which is expected for this prototype (no database).

---

## 3. What Needs to Be Done Next

1. **Manual User-Testing**:
   - Launch the portal and click through every page in the browser to verify visual presentation and interactions.
2. **Prepare Final Academic Deliverables (Before Nov 15, 2026)**:
   - **Written Project Report**: Document problem definition, dataset statistics, methodology, cross-validation comparison, SHAP analysis, and ethical implications.
   - **Presentation Slide Deck (PPT)**: Structure 8–10 slides for the oral defense (Problem $\rightarrow$ Data Checks $\rightarrow$ Model Results $\rightarrow$ SHAP Insights $\rightarrow$ EduGuard System $\rightarrow$ Live Demo).
   - **Project Archive (ZIP)**: Prepare clean ZIP containing source code and documentation for UMS submission.
3. **Optional Cloud Deployment**:
   - Push repository to GitHub and connect to [Streamlit Community Cloud](https://share.streamlit.io) to generate a public live demo URL.

---

## 4. Exact Commands to Restart Tomorrow

Run these commands in your PowerShell terminal from the project root:

### Step A: Activate Virtual Environment (if using one)
```powershell
# If using a virtual environment in .venv or venv:
.\venv\Scripts\Activate.ps1
```

### Step B: Install / Verify Dependencies
```powershell
pip install -r requirements.txt
```

### Step C: Run Pipeline Scripts (Only if artifacts are missing)
*Note: All artifacts already exist in the folder. You only need to run these if files were deleted:*
```powershell
# 1. Model cross-validation comparison (generates results.csv):
python train.py

# 2. Risk scoring, SHAP tree explanations, and artifacts (generates model.pkl, shap_values.npy, shap_summary.png):
python explain.py

# 3. Triage ranking verification (generates ranked_students.csv):
python briefing.py
```

### Step D: Launch the Web Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
