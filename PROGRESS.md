# Project Progress & Handoff Report
**Project Name:** EduGuard — Student Dropout Risk Prediction & Counselor Briefing System  
**Branch:** `website` (Full-Stack FastAPI + Vanilla JS/CSS/HTML Web Application)  
**Last Updated:** November 2, 2026  
**Target Submission Deadline:** November 15, 2026 (UMS Submission)  

---

## 1. What Has Been Completed So Far

### Core Machine Learning Pipeline (Main Branch / Fallback)
- **`data/data.csv`**: Loaded official UCI "Predict Students' Dropout and Academic Success" dataset (4,424 students, 36 features).
- **`train.py`**: Binary classification formulation (Dropout=1, Enrolled/Graduate=0) with Stratified 5-Fold Cross-Validation.
- **`results.csv`**: Empirical cross-validation metrics comparing Random Forest (ROC-AUC: 0.9198, Recall: 71.36%, Precision: 87.50%, F1: 78.59%) vs Logistic Regression.
- **`explain.py`**: Honest Out-of-Fold predicted probabilities (`risk_scores.csv`), production Random Forest (`model.pkl`), feature names (`features.pkl`), Class 1 SHAP matrix (`shap_values.npy`), and global plot (`shap_summary.png`).
- **`briefing.py`**: Situational urgency scoring, referral priority formula: $\text{Priority} = \text{Risk} \times (1 + \text{weight} \times \text{urgency})$, unscaled SHAP risk drivers extraction, and `NoRiskFactorsError` guardrail.
- **`assigner.py`**: 6 simulated advisors across Academic, Financial, and Engagement specialties (1,700 total capacity), feature-to-specialty mapper, and capacity-constrained auto-assignment algorithm.
- **`app.py`**: 7-page Streamlit portal preserved completely intact as a working fallback.

---

### Full-Stack Web Portal (Website Branch)

#### Step 1: FastAPI REST Backend (`server.py`)
- Created FastAPI backend instance `app` serving REST API endpoints and mounting `web/` static files:
  - `GET /api/summary`: High-level cohort KPIs, risk distribution, degree program risks, top 10 referrals.
  - `GET /api/students`: Filterable, searchable, sorted, paginated student directory with CSV export (`GET /api/export.csv`).
  - `GET /api/students/{id}`: Single student deep-dive, local SHAP feature attributions, unscaled real values, briefing text, and suggested actions.
  - `GET /api/counselors`: Roster of 6 simulated advisors with live caseloads, capacities, and utilization rates.
  - `POST /api/assign/auto`: Capacity-constrained greedy auto-assignment matching student SHAP drivers to specialist loads.
  - `POST /api/assign/{id}`: Manual advisor assignment/reassignment.
  - `POST /api/status/{id}`: Student outreach status updates (`Flagged`, `Contacted`, `Resolved`).
  - `POST /api/reset`: Reset all assignments to "Unassigned" and statuses to "Flagged".
  - `GET /api/metrics`: Cross-validation metrics, acceptance criteria verification matrix, and global feature importances.
- **Persistence (`state.json`)**: Persistent storage ensures that advisor assignments and student case statuses survive server reloads.

#### Step 2: Foundation, Hero Section & Dashboard (`web/`)
- Single-page application shell with Inter typography, navy/teal palette, collapsible sidebar, live urgency slider ($0.0 - 2.0$), and light/dark mode with `localStorage` persistence.
- Animated Landing Hero with gradient mesh, cohort overview stats, and "Open Dashboard" button.
- Dashboard Page (`📊 Dashboard`): 6 count-up KPI cards, Chart.js Risk Tier Doughnut, Average Risk by Course Bar Chart, and Top 10 Urgent Triage Referrals table.

#### Step 3: Students Directory (`👥 Students`)
- Multi-criteria filter toolbar: search by Student ID (debounced 300ms), filter by risk tier (`High`, `Med`, `Low`), assigned advisor, case status, sort field, and sort direction.
- Dynamic pagination with rows-per-page selector (15, 25, 50, 100) and boundary disabling.
- Quick advisor assignment modal dialog (`#assign-modal`) with dark-backdrop blur.
- Live filtered CSV export via `/api/export.csv`.

#### Step 4: Student Profile Deep-Dive (`👤 Student Profile`)
- Student Quick Switcher bar (`‹`, ID input, `Go`, `›`) to jump to any of the 4,424 students.
- Header KPI Card showing Student ID, Risk badge, Status pill, Historical outcome (`Dropout` vs `Retained`), Urgency score, Priority score, and Advisor.
- Interactive Local SHAP Horizontal Bar Chart (Chart.js) with red risk drivers, teal protective factors, and tooltips showing unscaled dataset values.
- Primary Risk Drivers cards with unscaled real values and domain badges (`Academic`, `Financial`, `Engagement`).
- Counselor Briefing box with `.txt` download button and `NoRiskFactorsError` guardrail warning alert.
- Case management actions: "⚡ Auto-Assign", "👤 Reassign Advisor", "📞 Mark Contacted", "✅ Mark Resolved".

#### Step 5: Counselors Roster (`🧑‍💼 Counselors`)
- Prominently displays the mandatory `Sample roster (simulated)` badge.
- Institutional overview cards: 6 active advisors, 1,700 seats, active assigned, available slots, and utilization percentage.
- Domain specialty filter tabs (`All`, `Academic`, `Financial`, `Engagement`).
- 3×2 counselor card grid with animated progress bars color-coded by load threshold (emerald $<75\%$, amber $75-90\%$, crimson $>90\%$).
- "👥 View Caseload →" deep-links directly into the Students Directory filtered to that advisor.

#### Step 6: Assignments Engine (`📋 Assignments`)
- "⚡ Run Auto-Assignment (High & Med)": processes 1,937 students, fills all 1,700 specialist slots, and designates 237 overflow students as `"Unassigned - capacity full"`.
- "↺ Reset All Assignments": cleans assignments and statuses back to defaults.
- Algorithmic triage card detailing the 4-step greedy constraint matching logic for viva defense.
- Live advisor caseload utilization table and top 10 priority triage queue preview table.

---

## 2. Bug Fixes & Refinements Applied

1. **NumPy Scalar Serialization (HTTP 500 Fix)**:
   - *Problem*: In `GET /api/students/{id}`, extracting unscaled feature values from `df_raw` returned `numpy.int64` and `numpy.float64` objects. FastAPI's `jsonable_encoder` failed to iterate over them and raised `TypeError: 'numpy.int64' object is not iterable`, resulting in HTTP 500.
   - *Fix*: Applied `.item()` in `server.py` to sanitize all numpy scalars into native Python `int` and `float` types. Tested across all 4,424 records with 100% success.
2. **Search Matching & Regex Error Handling**:
   - *Problem*: Searching with `#` or `Student #` prefixes failed exact digit matching, and `Series.str.contains()` defaulted to `regex=True`, throwing errors on regex characters.
   - *Fix*: Cleaned leading `#` and `Student` prefixes in `server.py` and `script.js`, and configured `str.contains(clean_term, regex=False)` so partial numbers and special characters are handled smoothly.

---

## 3. What Needs to Be Done Next

Only **2 steps remain** to complete the entire web application:

1. **Step 7: Reports Page (`📈 Reports`)**:
   - Stratified 5-Fold Cross-Validation metrics table from `results.csv`.
   - `ACCEPTANCE.md` Verification Matrix with live computed **PASS** badges for all 5 criteria.
   - Embed global SHAP feature importance plot (`shap_summary.png`).
   - Methodological Limitations callout box (Portuguese context, simulated roster, out-of-fold probabilities).
2. **Step 8: About Page (`ℹ️ About`) & Final Push**:
   - Project mission, 7-stage ML pipeline breakdown, and technology stack badges.
   - Final review and commit to GitHub branch `website`.

---

## 4. Exact Commands to Restart

From PowerShell in the project directory:

### Step A: Verify Dependencies
```powershell
pip install -r requirements.txt
```

### Step B: Launch the FastAPI Web Application
```powershell
uvicorn server:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser.

### (Fallback) Launch the Streamlit Prototype
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
