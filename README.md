# Student Dropout Risk Prediction & Counselor Briefing System

An end-to-end, explainable AI decision-support platform designed for higher education academic advisors and counselors. The system predicts student dropout risk, calculates situational operational urgency, prioritizes students in a triage queue, and generates individualized, explainable advising briefings using SHAP (SHapley Additive exPlanations).

---

## 📌 Project Architecture & Workflow

```text
       Raw Data (UCI Dataset)
                 │
                 ▼
       ┌──────────────────┐
       │     train.py     │ ──► Data Quality Checks & 5-Fold Stratified CV
       └──────────────────┘     (Logistic Regression Baseline vs Random Forest)
                 │              Outputs: results.csv
                 ▼
       ┌──────────────────┐
       │    explain.py    │ ──► Out-of-Fold Risk Scoring & SHAP TreeExplainer
       └──────────────────┘     Outputs: risk_scores.csv, shap_summary.png,
                 │                       model.pkl, features.pkl, shap_values.npy
                 ▼
       ┌──────────────────┐
       │   briefing.py    │ ──► Urgency Computation & Priority Ranking
       └──────────────────┘     Guardrail: NoRiskFactorsError
                 │              Outputs: ranked_students.csv
                 ▼
       ┌──────────────────┐
       │      app.py      │ ──► Interactive Streamlit Counselor Dashboard
       └──────────────────┘     (Triage Table, Urgency Slider, SHAP Charts, Briefings)
```

---

## 📁 Repository Structure

```text
AI PROJECT/
├── data/
│   └── data.csv              # Semicolon-delimited UCI student dataset (4,424 rows, 36 features)
├── ACCEPTANCE.md             # Measurable acceptance criteria & operational assumptions
├── requirements.txt          # Python library dependencies
├── train.py                  # Step 1: Data checks, binarization, Stratified 5-Fold CV
├── results.csv               # Model comparison results (mean ± std across 5 folds)
├── explain.py                # Step 2: OOF risk scores, final model, SHAP global summary
├── shap_summary.png          # Top 15 global feature importance bar plot
├── model.pkl                 # Persisted production Random Forest model
├── features.pkl              # Persisted list of feature column names
├── shap_values.npy           # Precomputed Class 1 (Dropout) SHAP matrix (4424 x 36)
├── risk_scores.csv           # Out-of-fold predicted dropout probabilities
├── briefing.py               # Step 3: Urgency calculation, referral priority, briefings
├── ranked_students.csv       # Triage queue ranked by referral priority
├── app.py                    # Step 4: Interactive Streamlit web application
├── VIVA_PREP.md              # Complete oral viva defense cheat sheet with Q&A
└── README.md                 # Project documentation (this file)
```

---

## 🚀 Installation & Setup

1. **Clone or navigate to the project directory:**
   ```powershell
   cd "c:\Users\gudiy\OneDrive\Documents\AI PROJECT"
   ```

2. **Install required dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify dataset location:**
   Ensure `data.csv` is placed inside the `data/` directory: `data/data.csv`.

---

## 💻 Execution Guide

Each module is modular and can be executed independently:

### Step 1: Data Quality Checks & 5-Fold CV Comparison
```powershell
python train.py
```
- Performs shape, missing value, duplicate, and class balance checks.
- Runs Stratified 5-Fold Cross-Validation on Logistic Regression (with StandardScaler pipeline) and Random Forest.
- Outputs summary table to console and saves to `results.csv`.

### Step 2: Out-of-Fold Risk Scoring & SHAP Explanations
```powershell
python explain.py
```
- Generates unbiased out-of-fold (OOF) predicted probabilities (`risk_scores.csv`).
- Trains final production Random Forest on the complete dataset (`model.pkl`).
- Applies SHAP `TreeExplainer` across all students (`shap_values.npy`).
- Saves global feature importance bar chart (`shap_summary.png`).

### Step 3: Triage Priority Ranking & Counselor Briefings
```powershell
python briefing.py
```
- Calculates situational urgency based on financial and academic crisis triggers.
- Computes Referral Priority scores and exports `ranked_students.csv`.
- Generates personalized briefings featuring actual student values and top 3 SHAP drivers.
- Tests the `NoRiskFactorsError` safety guardrail.

### Step 4: Launch the Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
- Opens the web application in your browser at `http://localhost:8501`.
- Adjust the **Urgency Weight** slider in the sidebar to modulate priority ranking in real time.
- Select any student from the triage queue to view their directional SHAP chart and download their advising briefing.

---

## 📊 Experimental Results

Evaluated using **Stratified 5-Fold Cross-Validation** on 4,424 students (1,421 Dropouts, 3,003 Retained):

| Model | Precision | Recall (Dropout) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 87.10% ± 2.08% | 72.27% ± 1.46% | 78.97% ± 0.97% | 0.9182 ± 0.0060 |
| **Random Forest (Final)** | **87.50% ± 2.67%** | **71.36% ± 0.49%** | **78.59% ± 1.08%** | **0.9198 ± 0.0100** |

Both models surpass the acceptance threshold established in `ACCEPTANCE.md` ($\text{ROC-AUC} \ge 0.85$, $\text{Precision} \ge 65\%$).

---

## 🧮 Core Methodological Formulas

### 1. Situational Urgency Score ($0.0 \text{ to } 1.0$)
Identifies immediate risk of administrative or academic dismissal:
$$\text{Urgency} = 0.4 \times (1 - \text{Tuition fees up to date}) + 0.3 \times \text{Debtor} + 0.3 \times \mathbb{I}_{(\text{approved}=0 \land \text{enrolled}>0)}$$

### 2. Referral Priority Formula
Combines statistical dropout risk with operational urgency:
$$\text{Referral Priority} = \text{Risk Score} \times (1 + \text{urgency\_weight} \times \text{urgency})$$
- When $\text{urgency\_weight} = 0$: Queue is purely probability-driven.
- When $\text{urgency\_weight} > 0$: Students facing imminent debt or academic disqualification deadlines jump to the top of the advising queue.

### 3. Top Risk Drivers (Local SHAP Attributions)
For any student $i$, the model ranks features where $\phi_{i,j} > 0$ (positive contributions to dropout risk) and extracts the student's actual unscaled feature value $x_{i,j}$. If $\forall j, \phi_{i,j} \le 0$, the system raises `NoRiskFactorsError`.
