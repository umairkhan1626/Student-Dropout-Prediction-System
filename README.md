# Student Dropout Risk Prediction & Counselor Briefing System (EduGuard)

An end-to-end, explainable AI decision-support platform designed for higher education academic advisors and counselors. The system predicts student dropout risk, calculates situational operational urgency, prioritizes students in a triage queue, and generates individualized, explainable advising briefings using SHAP (SHapley Additive exPlanations).

---

## 📁 Repository Structure

The project is cleanly decoupled into `backend/` and `frontend/` folders so each can be developed, tested, and deployed independently on different hosts:

```text
AI PROJECT/
├── backend/
│   ├── server.py              # FastAPI REST backend (API, CORS, state, and static serving fallback)
│   ├── briefing.py            # Step 3: Urgency calculation, referral priority, briefings
│   ├── assigner.py            # Specialty-matched counselor auto-assignment engine
│   ├── train.py               # Step 1: Data checks, binarization, Stratified 5-Fold CV
│   ├── explain.py             # Step 2: OOF risk scores, final model, SHAP global summary
│   ├── app.py                 # Interactive Streamlit counselor dashboard (fallback UI)
│   ├── styles.py              # Visual styling tokens for Streamlit
│   ├── requirements.txt       # Backend Python dependencies
│   ├── ACCEPTANCE.md          # Measurable acceptance criteria & operational assumptions
│   ├── results.csv            # Model CV comparison results
│   ├── model.pkl              # Persisted production Random Forest model
│   ├── features.pkl           # Persisted feature names list
│   ├── shap_values.npy        # Precomputed Class 1 SHAP values matrix (4424 x 36)
│   ├── risk_scores.csv        # Out-of-fold predicted dropout probabilities
│   ├── ranked_students.csv    # Triage queue ranked by referral priority
│   ├── state.json             # Persistent counselor assignments and student outreach statuses
│   ├── shap_summary.png       # Global feature importance summary plot
│   └── data/
│       └── data.csv           # Semicolon-delimited UCI student dataset (4,424 rows, 36 features)
├── frontend/
│   ├── index.html             # Single-page web application UI
│   ├── script.js              # Vanilla JS controller with dynamic API_BASE integration
│   ├── style.css              # Custom responsive stylesheet (Navy/Teal theme)
│   ├── config.js              # Configurable API_BASE URL setting
│   ├── results.csv            # Static copy of CV benchmark results
│   └── shap_summary.png       # Static copy of global SHAP summary plot
├── .gitignore                 # Git ignore rules (explicitly tracking required backend artifacts)
├── VIVA_PREP.md               # Oral viva defense cheat sheet with Q&A
└── README.md                  # Project documentation (this file)
```

---

## 🚀 Installation & Setup

1. **Clone or navigate to the project directory:**
   ```powershell
   cd "c:\Users\gudiy\OneDrive\Documents\AI PROJECT"
   ```

2. **Install backend dependencies:**
   ```powershell
   cd backend
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

### Method 1: Local Convenience (Single Command - Same Origin)
In development, the FastAPI backend automatically serves `frontend/` as static files if present:

```powershell
cd backend
uvicorn server:app --port 8000
```
Open **`http://localhost:8000`** in your browser. The full application (landing page, dashboard, student directory, counselor matching, reports) works immediately with zero configuration.

---

### Method 2: Running Backend & Frontend Separately (Decoupled Mode)
Use this mode to test true decoupled client-server behavior with cross-origin requests:

1. **Start the Backend:**
   ```powershell
   cd backend
   uvicorn server:app --port 8000
   ```

2. **Configure Frontend API URL:**
   Open `frontend/config.js` and ensure it points to the backend:
   ```javascript
   window.API_BASE = "http://localhost:8000";
   ```

3. **Start the Frontend on a Separate Port (e.g. 5500):**
   ```powershell
   cd frontend
   python -m http.server 5500
   ```
   Open **`http://localhost:5500`** in your browser. All API requests, export downloads, and auto-assignments communicate with `http://localhost:8000` via CORS.

---

### Method 3: Streamlit Fallback Dashboard
If you want to run the legacy Streamlit interface:
```powershell
cd backend
streamlit run app.py
```
Or from the project root:
```powershell
streamlit run backend/app.py
```
Opens in your browser at `http://localhost:8501`.

---

## 🌐 Production Deployment Guide

### Deploying the Backend (Python Host: Render, Railway, etc.)
- **Root Directory for Build:** `backend/`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:**
  ```bash
  uvicorn server:app --host 0.0.0.0 --port $PORT
  ```
- **Environment Variables:**
  - `PORT`: Set automatically by your host (defaults to `8000` locally).
  - `ALLOWED_ORIGINS`: Comma-separated list of allowed frontend domains (e.g. `https://my-eduguard-ui.vercel.app`), or `*` for public access.

### Deploying the Frontend (Static Host: Vercel, Netlify, Cloudflare Pages, GitHub Pages)
- **Publish Directory / Root:** `frontend/`
- **Configuration:**
  In `frontend/config.js`, update `window.API_BASE` to your deployed backend URL:
  ```javascript
  window.API_BASE = "https://your-backend-service.onrender.com";
  ```
- No build command required (pure HTML/CSS/Vanilla JS).

---

## 🔬 Model Training & Explainability Pipeline

If you ever wish to re-execute the training or explainability pipelines from scratch:

```powershell
cd backend

# Step 1: 5-Fold Stratified Cross-Validation
python train.py

# Step 2: Out-of-Fold Risk Scoring & SHAP TreeExplainer
python explain.py

# Step 3: Urgency Computation & Priority Ranking
python briefing.py
```

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
