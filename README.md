# BQIS — Biscuit Quality Intelligence System

> **Target audience of this document:** AI coding assistants, future developers, and auditors.
> This README is intentionally verbose and explicit so that any AI model can understand the
> codebase without needing to read every source file first.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Tech Stack](#2-tech-stack)
3. [Repository Structure](#3-repository-structure)
4. [Data Model and CSV Contract](#4-data-model-and-csv-contract)
5. [Backend — FastAPI (backend/main.py)](#5-backend--fastapi-backendmainpy)
6. [Frontend — React + Vite (frontend/)](#6-frontend--react--vite-frontend)
7. [ML Model Bundle](#7-ml-model-bundle)
8. [Risk Classification Logic](#8-risk-classification-logic)
9. [Known Issues and Active TODOs](#9-known-issues-and-active-todos)
10. [Environment Variables](#10-environment-variables)
11. [Running the Project Locally](#11-running-the-project-locally)
12. [ML Experiment History (Approaches 1–5)](#12-ml-experiment-history-approaches-15)
13. [SNI 2973:2022 Compliance Reference](#13-sni-29732022-compliance-reference)
14. [Design Decisions and Constraints](#14-design-decisions-and-constraints)

---

## 1. Project Purpose
BQIS is a biscuit quality audit dashboard and decision support system for certificate-style quality review under SNI 2973:2022. It combines a FastAPI backend, an XGBoost classifier, SHAP explainability, PCA-based failure clustering, PDF report generation, and a React dashboard for analysts, auditors, and management.
BQIS is a **food quality audit intelligence dashboard** built for biscuit manufacturers
The project is designed for batch analysis, not real-time sensing. You upload or load a CSV dataset, the backend scores each sample, assigns a risk level, groups failures into failure categories, and serves summary data to the frontend.
in collaboration with **TUV NORD** as the certification body.
## What The System Does

BQIS takes biscuit laboratory data and turns it into a structured quality review workflow.

It can:

- classify each sample as PASS or FAIL
- assign a risk level based on model probability
- identify whether a failure appears microbiological, physicochemical, heavy-metal related, or stability related
- summarize the dataset through dashboard KPIs and charts
- generate an audit report for technical reviewers
- generate an executive summary for non-technical stakeholders
- accept a new CSV upload and reprocess the full analysis pipeline

## High-Level Architecture

The application is split into three practical layers:

1. Data layer: CSV files in the data and versioned experiment folders.
2. Backend layer: FastAPI app in backend/main.py.
3. Frontend layer: React + Vite app in frontend/.

The backend loads the reference dataset and model artifacts on startup, computes predictions and explanations in memory, and exposes JSON endpoints. The frontend consumes those endpoints through a single API service module.

## Repository Layout

```text
BQIS/
|-- app.py                         # Legacy Streamlit app, kept for reference
|-- backend/
|   |-- main.py                    # FastAPI app, model pipeline, reports
|   |-- output.pdf                 # Example generated report artifact
|   `-- venv/                      # Local environment if present
|-- data/
|   |-- bqis_biscuit_quality_dataset.csv
|   `-- bqis_biscuit_quality_dataset_3000.csv
|-- frontend/
|   |-- package.json
|   |-- vite.config.js
|   |-- src/
|   |   |-- App.jsx
|   |   |-- main.jsx
|   |   |-- index.css
|   |   |-- components/
|   |   |-- constants/
|   |   |-- mock/
|   |   |-- pages/
|   |   `-- services/
|   `-- public/
|-- v1_baseline_full_features/
|-- v2_feature_selection/
|-- v3_robust_pipeline/
|-- v4_model_audit/
|-- v5_recalibration_sop/
`-- generate_dataset.py
```

The current backend code expects two runtime artifacts:

- v4_model_audit/bqis_model_bundle.pkl
- v2_feature_selection/bqis_clustering_result_v2.csv

If either file is missing, startup or downstream analysis will fail.

## Tech Stack

- Python
- FastAPI
- Uvicorn
- Pandas and NumPy for data processing
- scikit-learn for KNN imputation and PCA
- XGBoost for binary classification
- SHAP-style explainability via XGBoost contribution values
- fpdf2 for PDF report generation
- python-dotenv for environment loading
- Google GenAI client for optional narrative generation

### Frontend

- React 19
- Vite 8
- react-router-dom 7
- Axios for HTTP requests
- Recharts for charts
- Framer Motion for page motion and transitions
- react-icons for navigation icons
- Tailwind packages are installed, but the current UI styling is primarily in frontend/src/index.css

## Data Contract

The backend expects biscuit quality data to use exact, case-sensitive column names.

### Required Numeric Columns

These columns are used in the model pipeline and must be present or inferable in the uploaded CSV. If a column is missing entirely, the backend fills it with NaN and applies the missing-data policy.

| Column | Meaning |
|---|---|
| Moisture_Content_% | Moisture content |
| Fat_Content_% | Fat content |
| Protein_Content_% | Protein content |
| Water_Activity_Aw | Water activity |
| Acid_Insoluble_Ash_% | Acid insoluble ash |
| Acid_Value_mgKOHg | Acid value |
| Peroxide_Value | Peroxide value |
| Total_Plate_Count_CFUg | Total plate count |
| Yeast_Mold_Count_CFUg | Yeast and mold count |
| Lead_Pb_mgkg | Lead concentration |
| Cadmium_Cd_mgkg | Cadmium concentration |

### Required Categorical Column

| Column | Meaning |
|---|---|
| Product_Name | Product type, one-hot encoded before model inference |

### Optional Columns

| Column | Default behavior if missing |
|---|---|
| Sample_ID | Auto-generated as UPL-0001, UPL-0002, and so on |
| Batch_Code | Filled with BCH-UPLOADED |
| Test_Date | Parsed if present; otherwise period filtering is disabled |

## Missing Data Policy

The backend uses a row-level missing data threshold of 30% across the required numeric parameters.

- If a row has 30% or less missing values, it is kept and imputed with KNNImputer.
- If a row has more than 30% missing values, it is excluded from model scoring and marked for manual review.

Excluded rows receive:

- Prediction = N/A
- Risk_Level = Excluded - Manual Review
- Probability fields set to NaN

This is the core quality gate in the pipeline and should be preserved if the model is retrained or the dataset format changes.

## Backend Behavior

The backend entry point is backend/main.py.

### Startup Flow

When FastAPI starts, the lifespan hook calls load_data(). That function loads the baseline CSV from data/bqis_biscuit_quality_dataset.csv, processes it, computes model outputs, SHAP summaries, PCA coordinates, and cached chart data, and stores the results in module-level globals.

The main in-memory objects are:

- global_ds: processed dataset with predictions, risk levels, and data-quality flags
- global_feat: feature matrix used for the model
- global_bundle: loaded model bundle and feature column list
- global_shap_df: global feature importance summary
- global_pca_df: PCA coordinates plus failure category labels
- global_cluster_df: reserved / unused in the current code

Important detail: SHAP is computed once at startup and is not recomputed when filters change. Filtered views reuse the cached global SHAP ranking.

### Processing Pipeline

The main processing function follows a deterministic sequence:

1. Ensure all required numeric columns exist.
2. Auto-generate Sample_ID, Batch_Code, and Product_Name if needed.
3. Compute each row’s missing-value percentage.
4. Split the dataset into included and excluded rows using the 30% threshold.
5. Apply KNN imputation to included rows.
6. One-hot encode Product_Name and align the features with the model bundle.
7. Run XGBoost predictions and probabilities.
8. Assign PASS or FAIL.
9. Convert failure probability into High Risk, Medium Risk, Low Risk, or Pass.
10. Compute global feature importance from XGBoost contribution values.
11. Fit PCA for a two-dimensional cluster map.
12. Merge PCA results with the precomputed failure category CSV.
13. Mark excluded rows for manual review.
14. Restore original row order and derive Test_Date_dt for filtering.

### Risk Logic

The risk classification thresholds are:

- High Risk: probability of failure at or above 0.80
- Medium Risk: probability of failure at or above 0.60
- Low Risk: anything below the medium threshold, if the prediction is FAIL
- Pass: any sample predicted as PASS

## API Endpoints

All JSON endpoints live under /api and accept optional filter query parameters where applicable.

### Shared Filter Parameters

The following query parameters are supported by the dashboard-style endpoints:

- period: string in the format January 2025, February 2025, and so on
- batch: exact Batch_Code match
- product: exact Product_Name match

### GET /api/dashboard

Returns the main dashboard payload:

- total sample count
- pass and fail counts
- high-risk sample count
- average confidence
- pass and fail rates
- risk distribution for the donut chart
- top SHAP parameters
- up to 200 PCA scatter points

### GET /api/risk-overview

Returns:

- overall risk summary
- risk distribution
- a table of risk actions and priorities
- the 10 most recent samples in reverse chronological order

### GET /api/shap

Returns the full global importance ranking, including:

- parameter labels
- mean absolute SHAP values
- relative influence percentages
- positive or negative direction

This endpoint is not filter-aware because the model importance summary is global by design.

### GET /api/clusters

Returns the PCA scatter view and failure cluster summary:

- total cluster count
- dominant cluster
- affected sample count
- cluster profiles for the four failure categories
- variance explanation placeholders for PC1 and PC2

### GET /api/executive-summary

Returns a business-oriented summary for management:

- pass/fail totals
- average confidence
- pass and fail rates
- risk summary counts
- top risk categories
- parameter impact summary
- a short audit recommendation sentence

### GET /api/filters/options

Returns the available filter dropdown values:

- periods derived from Test_Date
- batches derived from Batch_Code
- products derived from Product_Name

### POST /api/upload

Accepts a CSV file upload in multipart/form-data under the field name file.

Validation rules:

- only .csv files are accepted
- the file must be readable by pandas
- required columns must be present and case-sensitive
- empty files are rejected

On success, the backend reprocesses the dataset and returns a short quality summary.

### GET /api/report/audit

Returns a PDF download for technical reviewers.

The report includes:

- executive summary table
- risk distribution table
- top SHAP parameters
- failure cluster profiles
- missing-data handling notes
- methodology reference
- an audit recommendation section

### GET /api/report/executive

Returns a PDF download for non-technical readers.

The report is written in more business-friendly language and emphasizes:

- key findings
- risk breakdown
- recommended action

## Frontend Structure

The frontend lives in frontend/ and is a React single-page application built with Vite.

### Routes

The app routes are defined in frontend/src/App.jsx:

- / - Dashboard
- /data-input - CSV upload and dataset ingestion
- /sample-risk - sample-level risk overview
- /failure-pattern - PCA and failure-pattern view
- /parameter-importance - full SHAP ranking
- /executive-summary - management summary and report access

The sidebar mirrors the same destinations and uses the same navigation model.

### Page Responsibilities

Each page is narrow in scope:

- Dashboard combines KPIs, donut charts, SHAP highlights, and a PCA snapshot.
- Data Input handles CSV upload and reprocessing.
- Sample Risk Overview focuses on sample-level risk distribution and recent cases.
- Failure Pattern shows cluster maps and failure grouping.
- Parameter Importance expands the SHAP view into a full ranked list.
- Executive Summary presents business-level messaging and report generation links.

### Shared UI Components

The component layer under frontend/src/components/ includes reusable pieces for:

- sidebar navigation
- page header and breadcrumbs
- filter bar controls
- stat cards
- donut charts
- SHAP charts
- scatter plot charts
- data tables
- insight and recommendation cards

### API Service Layer

All backend communication is centralized in frontend/src/services/api.js.

The service exposes functions for:

- fetching filter options
- loading dashboard data
- loading risk overview data
- loading SHAP data
- loading cluster data
- loading executive summary data
- downloading PDF reports
- uploading CSV files

The service currently supports mock JSON payloads in frontend/src/mock/, but the code is configured for live backend use by default.

### Styling And UI

The UI is styled primarily with frontend/src/index.css.

The current design system is built around:

- a dark navy sidebar
- white content surfaces
- card-based analytics panels
- consistent chart colors shared with the backend
- subtle motion on page transitions

## Frontend And Backend Connection

Vite proxies /api requests to the Python backend at http://127.0.0.1:8000.

That means a typical local development flow is:

- run FastAPI on port 8000
- run Vite on port 5173
- let the frontend call /api/* through the proxy

## Color And Category Rules

Failure categories use a fixed shared palette.

The canonical colors are:

- Microbiological: #E74C3C
- Physicochemical: #3498DB
- Heavy_Metal: #9B59B6
- Stability: #F5B041
- Pass: #2ECC71

These values are used both in the frontend and in the backend PDF/report logic. If you change them, update both sides together.

## Local Setup

The repository appears to be structured for local development on Windows, but the commands below are standard for any platform.

### 1. Backend

From the repository root:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If the project does not yet have a requirements.txt file, install the backend dependencies that are imported by backend/main.py.

### 2. Frontend

From frontend/:

```bash
npm install
npm run dev
```

### 3. Open The App

After both services are running, open the Vite URL shown in the terminal, usually http://127.0.0.1:5173.

## Environment Variables

The backend optionally reads:

- GOOGLE_AI_API_KEY

If the key is present, the report generator can call Gemini for narrative text. If it is absent or invalid, the backend falls back to local template text and still works.

## Dataset And Experiment Folders

The versioned folders in the repository document the modeling history of the project.

- v1_baseline_full_features contains the baseline notebook work.
- v2_feature_selection contains feature-selection and clustering work, plus the cluster label CSV used at runtime.
- v3_robust_pipeline contains the production-ready pipeline notebooks.
- v4_model_audit contains model-audit work and the model bundle expected by the backend.
- v5_recalibration_sop contains recalibration and operating-procedure work.

These folders are useful for auditability. They explain how the model evolved and why the current runtime pipeline looks the way it does.

## Practical Notes And Constraints

- The backend is stateful in memory. Restarting the server reloads the baseline dataset and recomputes derived outputs.
- The frontend expects the backend to be available through the Vite proxy unless mock mode is deliberately enabled.
- CSV uploads must use exact column names. Case mismatches will produce validation errors.
- The failure clusters are not discovered dynamically at runtime. They depend on the precomputed cluster-label CSV.
- SHAP values are global model characteristics, not a per-filter recalculation.
- The app is intended for batch analytics and audit review, not live process control.

## Known Gaps To Be Aware Of

- The repository currently mixes a few historical approaches and generated artifacts. That is intentional, but it makes the folder structure richer than a minimal production app.
- Some report text in the backend may still use older phrasing from earlier experimentation. The architecture, however, consistently targets the current BQIS workflow.
- If the model bundle or cluster-label CSV are absent, the backend cannot fully initialize.

## Why This README Is Verbose

This project is used in an audit-style context. The README is intentionally detailed so a future developer, reviewer, or AI assistant can understand:

- what the system does
- where each responsibility lives
- which files are operationally important
- how data must be shaped before it enters the pipeline
- how the backend and frontend are connected

That is the minimum documentation needed to modify the project safely.
- Generates two types of **PDF audit reports**: technical (for auditors) and executive (for management)
- Provides an interactive React dashboard with 6 pages

**It is NOT a real-time sensor system.** Data is batch-uploaded via CSV or the system reads
a static dataset on startup.

---

## 2. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend | Python + FastAPI | uvicorn server |
| ML Model | XGBoost (binary classifier) | Loaded from .pkl bundle |
| Explainability | SHAP (TreeExplainer via `booster.predict(pred_contribs=True)`) | Mean absolute SHAP values |
| Clustering | PCA (2D) + pre-labeled CSV | K-Means results from Approach 2 |
| Missing Data | KNN Imputer (sklearn) | n_neighbors=5, threshold=30% |
| PDF Reports | fpdf2 (FPDF) | Custom `BQISReport` subclass |
| AI Narrative | Google Gemini (`gemini-2.5-flash`) | Falls back to template if no API key |
| Frontend | React 18 + Vite | |
| Routing | React Router v6 | |
| Charts | Recharts | Bar, Donut (Pie), Scatter |
| Animation | Framer Motion | `fadeIn` shared variant |
| HTTP Client | Axios | Proxy: `/api` → `http://localhost:8000` |
| Styling | Vanilla CSS (`index.css`) | No Tailwind |

---

## 3. Repository Structure

```
BQIS/
|
+-- backend/
|   +-- main.py                  # ENTIRE backend: FastAPI app, ML pipeline, PDF gen
|
+-- frontend/
|   +-- src/
|   |   +-- App.jsx              # Root router (BrowserRouter + 6 routes)
|   |   +-- main.jsx             # Vite entry point
|   |   +-- index.css            # All CSS (CSS variables, layout, components)
|   |   +-- pages/
|   |   |   +-- Dashboard.jsx           # Page 1: Overview KPIs + SHAP + Scatter
|   |   |   +-- DataInput.jsx           # Page 2: CSV upload with drag-drop
|   |   |   +-- SampleRiskOverview.jsx  # Page 3: Risk table + donut chart
|   |   |   +-- FailurePattern.jsx      # Page 4: PCA scatter + cluster detail
|   |   |   +-- ParameterImportance.jsx # Page 5: Full SHAP bar chart
|   |   |   +-- ExecutiveSummary.jsx    # Page 6: KPIs + AI rec + PDF download
|   |   +-- components/
|   |   |   +-- Sidebar.jsx             # Navigation sidebar (6 links)
|   |   |   +-- Header.jsx              # Page title + breadcrumbs
|   |   |   +-- FilterBar.jsx           # Period/Batch/Product dropdowns + Apply button
|   |   |   +-- StatCard.jsx            # Single metric card
|   |   |   +-- DonutChart.jsx          # Recharts PieChart wrapper
|   |   |   +-- ShapChart.jsx           # SHAP horizontal bar chart (Recharts)
|   |   |   +-- ScatterPlotChart.jsx    # PCA scatter (Recharts ScatterChart)
|   |   |   +-- DataTable.jsx           # Generic sortable table
|   |   |   +-- InsightCard.jsx         # Narrative insight box
|   |   |   +-- RecommendationCard.jsx  # Highlighted recommendation box
|   |   +-- constants/
|   |   |   +-- colors.js               # SINGLE SOURCE OF TRUTH for all colors
|   |   |   +-- animations.jsx          # fadeIn variant + LoadingSkeleton components
|   |   +-- services/
|   |   |   +-- api.js                  # All API calls (Axios) + mock toggle
|   |   +-- mock/                       # Static JSON for mock mode
|   |       +-- dashboard.json / risk.json / shap.json / clusters.json / summary.json
|   +-- index.html
|   +-- vite.config.js               # Proxy: /api -> http://localhost:8000
|   +-- package.json
|
+-- data/
|   +-- bqis_biscuit_quality_dataset.csv        # Primary dataset (1000 rows) [loaded on startup]
|   +-- bqis_biscuit_quality_dataset_3000.csv   # Extended dataset (3000 rows)
|
+-- v1_baseline_full_features/        # Approach 1 notebooks (archived)
+-- v2_feature_selection/
|   +-- bqis_clustering_result_v2.csv   # *** REQUIRED AT RUNTIME — cluster labels ***
|   +-- BQIS_03_FeatureSelected_MultiMethod_Clustering.ipynb
+-- v3_robust_pipeline/               # Approach 3 notebooks
+-- v4_model_audit/
|   +-- bqis_model_bundle.pkl          # *** REQUIRED AT RUNTIME — XGBoost model ***
|   +-- BQIS_05_Model_Audit_Simulator.ipynb
+-- v5_recalibration_sop/             # Approach 5 notebooks
|
+-- generate_dataset.py               # Script to regenerate dummy dataset
+-- app.py                            # OLD Streamlit app (archived, NOT in use)
+-- .env                              # GOOGLE_AI_API_KEY
+-- README.md                         # This file
```

**Two files are REQUIRED at runtime.** Backend crashes on startup if either is missing:
1. `v4_model_audit/bqis_model_bundle.pkl` — XGBoost classifier + feature column list
2. `v2_feature_selection/bqis_clustering_result_v2.csv` — pre-computed cluster labels per Sample_ID

---

## 4. Data Model and CSV Contract

### Required Columns (all numeric unless noted; names are CASE-SENSITIVE)

| Column Name | Type | SNI 2973:2022 Limit | Notes |
|---|---|---|---|
| `Moisture_Content_%` | float | max 5.0% | Leading SHAP driver |
| `Fat_Content_%` | float | — | |
| `Protein_Content_%` | float | — | |
| `Water_Activity_Aw` | float | — | 2nd SHAP driver |
| `Acid_Insoluble_Ash_%` | float | max 0.1% | |
| `Acid_Value_mgKOHg` | float | max 2.0 mg KOH/g | Was 1.0 in dummy generator — corrected in Approach 4 |
| `Peroxide_Value` | float | NOT in SNI 2973:2022 | ML feature only, not an official SNI parameter |
| `Total_Plate_Count_CFUg` | float | Tiered by product type | Dataset uses single generic limit |
| `Yeast_Mold_Count_CFUg` | float | Tiered by product type | Dataset uses single generic limit |
| `Lead_Pb_mgkg` | float | max 0.50 mg/kg | |
| `Cadmium_Cd_mgkg` | float | max 0.20 mg/kg | |
| `Product_Name` | string | — | One-hot encoded (e.g. "Butter Biscuit", "Marie Biscuit") |

### Optional Columns (auto-generated if absent)

| Column Name | Default if Missing | Effect if Missing |
|---|---|---|
| `Sample_ID` | `UPL-0001`, `UPL-0002`, … | Generic IDs assigned |
| `Batch_Code` | `BCH-UPLOADED` | Batch filter won't match any specific batch |
| `Test_Date` | `NaT` | Period filter is disabled |

### Failure Category Labels (from `bqis_clustering_result_v2.csv`)

Column `Failure_Category_Original` maps each `Sample_ID` to one of:

| Internal Key | Display Label | Risk Level | Hex Color |
|---|---|---|---|
| `Microbiological` | Microbiology | High | `#E74C3C` |
| `Physicochemical` | Chemical | High | `#3498DB` |
| `Heavy_Metal` | Heavy Metal | High | `#9B59B6` |
| `Stability` | Moisture / Stability | Medium | `#F5B041` |
| `Pass` | Pass | — | `#2ECC71` |

> **AI RULE:** These colors are the **single authoritative palette** defined in
> `frontend/src/constants/colors.js`. The backend uses the same values as RGB tuples
> in `main.py` lines 717–723. **Never hardcode category colors outside these two files.**

---

## 5. Backend — FastAPI (backend/main.py)

### 5.1 Startup and Global State

On startup (FastAPI `lifespan` context manager), `load_data()` reads
`data/bqis_biscuit_quality_dataset.csv` and runs `process_dataset()`.

**Six global variables** are populated once and reused by all endpoints:

| Variable | Type | Contents |
|---|---|---|
| `global_ds` | `pd.DataFrame` | Full processed dataset with all predictions and risk levels |
| `global_feat` | `pd.DataFrame` | One-hot encoded feature matrix used for model input |
| `global_bundle` | `dict` | `{"model": XGBClassifier, "feature_columns": list[str]}` |
| `global_shap_df` | `pd.DataFrame` | Global SHAP importance (NOT re-filtered per request) |
| `global_pca_df` | `pd.DataFrame` | PCA (PC1, PC2) + `Failure_Category_Original` per sample |
| `global_cluster_df` | unused | Reserved |

> **AI NOTE:** `global_shap_df` is computed **once on startup** and is **NOT re-computed
> when filters change.** SHAP represents full dataset model behavior — only counts/
> distributions change when filters are applied via `apply_filters()`.

### 5.2 Processing Pipeline

`process_dataset(raw_df)` — called on startup and after every CSV upload:

```
Step 1   Ensure all 11 numeric columns exist (fill NaN if column entirely absent)
Step 2   Compute per-row missing % → split included (<=30%) vs excluded (>30%)
Step 3   KNN imputation (n_neighbors=5, auto-reduced for small datasets) on included rows only
Step 4   One-hot encode Product_Name → reindex to feature_columns → XGBoost predict
Step 5   Assign Prediction ("PASS"/"FAIL"), Prob_Pass, Prob_Fail, Confidence per row
Step 6   Assign Risk_Level per row (see Section 8)
Step 7   SHAP: booster.predict(pred_contribs=True) → mean|SHAP| per feature → shap_df
Step 8   PCA (2 components) on feature matrix → pca_df
Step 9   Merge pca_df with bqis_clustering_result_v2.csv on Sample_ID
         Fallback if merge fails: assign category from PC1-based modulo heuristic
Step 10  Mark excluded rows: Prediction="N/A", Risk_Level="Excluded - Manual Review"
Step 11  Merge included + excluded, sort by original index
Step 12  Parse Test_Date → Test_Date_dt column (enables period filter)
```

**Missing data policy (BQIS Proposal standard):**
- `>30%` missing params per row → excluded from ML, `Risk_Level = "Excluded - Manual Review"`
- `<=30%` missing params per row → KNN imputed, then processed normally
- Constant: `MISSING_THRESHOLD = 0.30` (line 86 in `main.py`)

### 5.3 All API Endpoints

**Filter-aware endpoints** accept optional query params:
- `period` (string, e.g. `"January 2025"`) — matches year+month of `Test_Date`
- `batch` (string, e.g. `"BCH-07-003"`) — exact match on `Batch_Code`
- `product` (string, e.g. `"Butter Biscuit"`) — exact match on `Product_Name`

Filters are applied via `apply_filters(global_ds, global_pca_df, period, batch, product)`
which returns **filtered copies** without mutating global state.

---

#### `GET /api/dashboard`
**Purpose:** Main dashboard page. KPIs, risk donut, SHAP top 5, PCA scatter (max 200 pts sampled).

**Response shape:**
```json
{
  "totalSamples": 1000,
  "predictedPass": 650,
  "predictedFail": 350,
  "highRiskSamples": 200,
  "avgConfidence": 91.3,
  "passRate": 65.0,
  "failRate": 35.0,
  "riskDistribution": [
    {"name": "Pass",        "value": 650, "color": "#2ECC71"},
    {"name": "High Risk",   "value": 200, "color": "#E74C3C"},
    {"name": "Medium Risk", "value": 100, "color": "#F5B041"},
    {"name": "Low Risk",    "value": 50,  "color": "#82E0AA"}
  ],
  "topShap": [
    {"label": "Moisture Content", "meanAbs": 0.412, "relativePct": 22.1}
  ],
  "scatterPoints": [
    {"id": "S001", "pc1": 1.23, "pc2": -0.45, "cluster": "Microbiological"}
  ]
}
```

---

#### `GET /api/risk-overview`
**Purpose:** Sample Risk Overview page. Risk summary, donut, priority table, recent 10 samples.

**Response shape:**
```json
{
  "totalSamples": 1000,
  "predictedPass": 650,
  "predictedFail": 350,
  "highRisk": 200,
  "avgConfidence": 91.3,
  "riskDistribution": [...],
  "riskTable": [
    {"level": "Pass",        "count": 650, "pct": "65.0%", "action": "Clear for certification",  "priority": "—"},
    {"level": "High Risk",   "count": 200, "pct": "20.0%", "action": "Immediate auditor review", "priority": "URGENT"},
    {"level": "Medium Risk", "count": 100, "pct": "10.0%", "action": "Selective re-testing",     "priority": "HIGH"},
    {"level": "Low Risk",    "count": 50,  "pct": "5.0%",  "action": "Standard monitoring",      "priority": "NORMAL"}
  ],
  "recentSamples": [
    {"id": "S001", "batch": "BCH-07-003", "product": "Butter Biscuit",
     "prediction": "FAIL", "risk": "High Risk", "confidence": 94.2}
  ]
}
```

---

#### `GET /api/shap`
**Purpose:** Full SHAP chart (all features). **Not filter-aware** — always global model behavior.

**Response shape:**
```json
{
  "paramCount": 13,
  "affectedSamples": 1000,
  "parameters": [
    {"label": "Moisture Content", "meanAbs": 0.412, "relativePct": 22.1, "direction": "Positive"}
  ]
}
```

---

#### `GET /api/clusters`
**Purpose:** Failure Pattern page. Cluster counts and PCA scatter points (filter-aware).

**Response shape:**
```json
{
  "totalClusters": 4,
  "dominantCluster": "Microbiological",
  "highRiskClusters": 2,
  "affectedSamples": 396,
  "scatterPoints": [
    {"id": "S001", "pc1": 1.23, "pc2": -0.45, "cluster": "Microbiological"}
  ],
  "clusterProfiles": [
    {"key": "Microbiological", "samples": 142},
    {"key": "Physicochemical", "samples": 87},
    {"key": "Heavy_Metal",     "samples": 67},
    {"key": "Stability",       "samples": 100}
  ],
  "varExp": {"pc1": 38.2, "pc2": 15.6}
}
```

> **ISSUE-003:** `varExp` is **hardcoded**. It does not recalculate on filter or upload.

---

#### `GET /api/executive-summary`
**Purpose:** Executive Summary page. KPIs, risk summary grid, top risks table, SHAP top 5.

**Response shape:**
```json
{
  "totalSamples": 1000,
  "predictedPass": 650,
  "predictedFail": 350,
  "avgConfidence": 91.3,
  "passRate": 65.0,
  "failRate": 35.0,
  "riskSummary": {"high": 200, "medium": 100, "low": 50, "pass": 650},
  "topRisks": [
    {"category": "Microbiological Contamination", "samples": 142,
     "risk": "High", "action": "Immediate re-inspection"}
  ],
  "parameterImpact": [
    {"parameter": "Moisture Content", "shapVal": 0.412}
  ],
  "auditRecommendation": "High risk alerts detected for 200 samples. Immediate quarantine recommended for flagged batches."
}
```

> **ISSUE-001:** `auditRecommendation` is **hardcoded**. Does not adapt to dominant failure
> category or active filter. See Section 9 for planned fix.

---

#### `GET /api/filters/options`
**Purpose:** Populate FilterBar dropdowns on page load.

**Response shape:**
```json
{
  "periods": ["January 2025", "February 2025", "March 2025"],
  "batches": ["BCH-06-012", "BCH-07-001", "BCH-07-003"],
  "products": ["Butter Biscuit", "Marie Biscuit"]
}
```

---

#### `POST /api/upload`
**Purpose:** Upload a new CSV dataset. **Replaces all global data** after processing.

**Request:** `multipart/form-data`, field `file`, `.csv` extension only.

**Validation steps (in order, returns HTTP error on first failure):**
1. File extension must be `.csv` → HTTP 400
2. CSV must be parseable by `pd.read_csv()` → HTTP 422
3. CSV must not be empty → HTTP 422
4. All 12 required columns must be present (case-sensitive) → HTTP 422
5. `process_dataset()` must succeed → HTTP 500

**Success response:**
```json
{
  "message": "Dataset berhasil diupload dan diproses",
  "samples": 1000,
  "quality_summary": {
    "analyzed": 950,
    "excluded_manual_review": 50,
    "threshold_pct": 30
  }
}
```

---

#### `GET /api/report/audit`
**Purpose:** Generate and stream a **technical PDF Audit Report** for auditors and lab analysts.

**PDF sections:**
1. Cover page (filter metadata + audience label)
2. Executive Summary (metrics table with contextual notes from `_executive_notes()`)
3. Risk Distribution (colored table — red/yellow/green rows)
4. Top 8 SHAP Parameters (XGBoost explainability table with relative %)
5. Failure Cluster Profiles (sorted by sample count descending)
6. Data Quality and Missing Value Handling
7. Methodology Reference (5 bullet points: XGBoost, SHAP, PCA, KNN, SNI)
8. Audit Recommendation (SHAP-driven, dominant cluster highlighted)

**Response:** `application/pdf`, `Content-Disposition: attachment; filename=audit_report.pdf`

---

#### `GET /api/report/executive`
**Purpose:** Generate and stream a **plain-language Executive Summary PDF** for management/clients.

**PDF sections:**
1. Cover page
2. Key Findings (narrative paragraph with pass/fail counts)
3. Risk Breakdown (2-column table)
4. Primary Quality Concerns (plain-label concern areas, sorted by count)
5. **Recommended Next Steps** — AI-generated via Gemini (`"next_steps"` section)
6. **Certification Impact** — AI-generated via Gemini (`"certification_impact"` section)

**Response:** `application/pdf`, `Content-Disposition: attachment; filename=executive_summary.pdf`

---

### 5.4 PDF Report Generation

Both reports use `class BQISReport(FPDF)` (defined at `main.py` line 726).

**Brand colors (RGB tuples in `main.py` lines 717–723):**
```python
NAVY   = (0, 32, 91)      # TUV NORD blue — headers, titles
RED    = (231, 76, 60)    # High risk rows
YELLOW = (245, 176, 65)   # Medium risk rows
GREEN  = (46, 204, 113)   # Pass rows
LIGHT  = (248, 249, 250)  # Background fills
GREY   = (127, 140, 141)  # Footer, secondary text
DARK   = (51, 51, 51)     # Body text
```

**Methods:**
- `header()` — Skipped on page 1 (cover). Pages 2+: "BQIS Audit Report" + page number.
- `footer()` — Skipped on page 1. Pages 2+: "BQIS -- TUV NORD | Generated: YYYY-MM-DD HH:MM"
- `h1(text)` — Section title (16pt bold navy)
- `h2(text)` — Subsection title (14pt bold dark)
- `_cover(subtitle, period, batch, product, audience_label)` — Cover page with navy banner
- `_table(headers, rows, col_widths, aligns, row_colors)` — Bordered table, navy header row

> **AI NOTE:** `fpdf2` uses **latin-1 encoding** by default. All unicode characters in
> generated text must be sanitized before passing to `multi_cell()`:
> `safe_text = text.encode('latin-1', 'replace').decode('latin-1')`
> This is already applied to AI-generated text but must be added to any new user-facing strings.

### 5.5 AI Narrative Generation (Gemini)

`generate_ai_narrative(context: dict, section: str) -> str`
- Model: `gemini-2.5-flash` via `google.genai.Client`
- Output: 2–3 sentence plain English paragraph (no markdown, no bullet points)

**Sections:**
- `"next_steps"` → actionable recommendation for management before re-certification
- `"certification_impact"` → business/timeline impact if issues are not resolved

**Context dict keys:**
```python
{
  "total":              int,    # total samples in filtered view
  "n_fail":             int,    # predicted FAIL count
  "fail_pct":           float,  # fail percentage
  "period":             str,    # filter period ("All Time" if unset)
  "batch":              str,    # filter batch ("All Batches" if unset)
  "batch_label":        str,    # human label ("BCH-07-003" or "the flagged batches")
  "product":            str,    # filter product
  "top_category_label": str,    # plain-language dominant failure category
  "top_category_count": int     # sample count for dominant category
}
```

**Fallback (no `GOOGLE_AI_API_KEY` or API error):**
```
next_steps:
  "We recommend {batch_label} be placed on hold pending corrective action.
   Production teams should review {top_category_label} control points before
   resubmission for certification."

certification_impact:
  "At current compliance rates, an estimated {n_fail} batches ({fail_pct:.1f}%)
   may face certification delays if underlying quality issues are not addressed
   prior to re-testing."
```

> **ISSUE-002:** The fallback is **generic across all failure categories**. It should provide
> category-specific corrective actions. See Section 9 for fix plan.

---

## 6. Frontend — React + Vite (frontend/)

### 6.1 Routing and Pages

Defined in `frontend/src/App.jsx`. Layout: fixed `<Sidebar>` (left) + scrollable
`<main class="layout-content">` (right).

| Route | Component | API Endpoint | Key Content |
|---|---|---|---|
| `/` | `Dashboard.jsx` | `GET /api/dashboard` | 4 KPIs, risk donut, SHAP bar, PCA scatter |
| `/data-input` | `DataInput.jsx` | `POST /api/upload` | Drag-drop CSV upload, column reference card |
| `/sample-risk` | `SampleRiskOverview.jsx` | `GET /api/risk-overview` | Risk donut, priority table, recent samples |
| `/failure-pattern` | `FailurePattern.jsx` | `GET /api/clusters` | PCA scatter (colored by cluster), cluster detail panel |
| `/parameter-importance` | `ParameterImportance.jsx` | `GET /api/shap` | Full SHAP bar chart + direction table |
| `/executive-summary` | `ExecutiveSummary.jsx` | `GET /api/executive-summary` | KPIs, risk grid, AI recommendation, PDF buttons |

**Common page pattern:**
```jsx
const [data, setData] = useState(null)
// On mount and on filter apply → fetch API → setData(r.data)
// Render: if (!data) return <LoadingSkeleton variant="..." />
// Root element: <motion.div {...fadeIn}>...</motion.div>
```

### 6.2 API Service Layer (frontend/src/services/api.js)

**Key constant:**
```js
const USE_MOCK = false  // Set true to use local JSON mock files (no backend needed)
```

When `USE_MOCK = false`: Axios `baseURL = '/api'`, proxied by Vite to `http://localhost:8000`.

**All exported functions:**
| Function | Method | Endpoint |
|---|---|---|
| `fetchFilterOptions()` | GET | `/api/filters/options` |
| `fetchDashboard(params)` | GET | `/api/dashboard` |
| `fetchRiskOverview(params)` | GET | `/api/risk-overview` |
| `fetchShap()` | GET | `/api/shap` |
| `fetchClusters(params)` | GET | `/api/clusters` |
| `fetchExecutiveSummary(params)` | GET | `/api/executive-summary` |
| `downloadReport(type, params)` | GET | `/api/report/{type}` (responseType: `'blob'`) |
| `uploadDataset(file)` | POST | `/api/upload` (multipart/form-data) |

**`params` shape for filter-aware calls:**
```js
{ period: "January 2025", batch: "BCH-07-003", product: "Butter Biscuit" }
```

### 6.3 Shared Constants

#### `frontend/src/constants/colors.js` — **SINGLE SOURCE OF TRUTH for all colors**

```js
FAILURE_COLORS   // { Microbiological, Physicochemical, Heavy_Metal, Stability, Pass } — hex strings
FAILURE_LABELS   // { Microbiological: "Microbiology", Physicochemical: "Chemical", ... }
RISK_COLORS      // { "High Risk", "Medium Risk", "Low Risk", "Pass", + short key aliases }
RISK_BG          // Pastel background colors for badges and cards
RISK_BORDER      // Pastel border colors
PDF_COLORS       // RGB tuples for reference (must match backend main.py lines 717-723)
```

> **Rule:** All components MUST import colors from this file. Do NOT hardcode hex values
> in component files. See ISSUE-004 for a known violation in `ExecutiveSummary.jsx`.

#### `frontend/src/constants/animations.jsx`

```js
fadeIn           // Framer Motion variant: opacity 0→1, y 8→0, duration 0.2s
LoadingSkeleton  // React component, prop: variant ("dashboard"|"risk"|"shap"|"cluster"|"exec")
```

### 6.4 Component Reference

| Component | Key Props | Notes |
|---|---|---|
| `Sidebar` | none | Fixed left nav, 6 links with react-icons |
| `Header` | `breadcrumbs[]`, `title`, `subtitle` | Page title block at top of every page |
| `FilterBar` | `period`, `batch`, `product`, `onPeriodChange`, `onBatchChange`, `onProductChange`, `onApply`, `buttonLabel` | Fetches `/api/filters/options` on mount to populate dropdowns |
| `StatCard` | `label`, `value`, `desc`, `accent` | Single metric tile |
| `DonutChart` | `data[]`, `title`, `height` | Recharts PieChart with center label + legend |
| `ShapChart` | `data[]`, `height` | Horizontal bar chart, colored by SHAP direction |
| `ScatterPlotChart` | `data[]`, `height` | Recharts ScatterChart, colored by cluster key |
| `DataTable` | `columns[]`, `rows[]` | Generic table |
| `InsightCard` | `title`, `text` | Styled insight box |
| `RecommendationCard` | `text` | Warning-colored recommendation box |

---

## 7. ML Model Bundle

**File:** `v4_model_audit/bqis_model_bundle.pkl` (loaded via `joblib.load`)

**Contents:**
```python
{
  "model":           XGBClassifier,  # Binary classifier: PASS=0, FAIL=1
  "feature_columns": list[str],      # One-hot expanded names (includes Product_ dummies)
}
```

**Prediction flow:**
```python
preds  = model.predict(df_feat)          # array of 0 or 1
probas = model.predict_proba(df_feat)    # shape (n, 2): [:, 0]=P(PASS), [:, 1]=P(FAIL)
```

**SHAP extraction:**
```python
booster   = model.get_booster()
contribs  = booster.predict(xgb.DMatrix(df_feat), pred_contribs=True)
shap_vals = contribs[:, :-1]   # last column is bias term, excluded
mean_abs  = np.abs(shap_vals).mean(axis=0)
```

**Model performance (Approach 4 audit, dummy dataset):**
- 5-fold CV accuracy: 0.9638 ± 0.0025
- Held-out test (20%) accuracy: 0.9800
- Held-out F1-Score: 0.9710

> **CRITICAL DISCLAIMER:** Partial data leakage exists (~19pp inflation). Model is trained
> on synthetic data and is NOT validated on real factory data. This must be disclosed
> every time accuracy figures are presented. See Section 12 for details.

---

## 8. Risk Classification Logic

Applied per row in `process_dataset()` after XGBoost prediction:

```python
RISK_THRESHOLD_HIGH   = 0.80  # Prob_Fail >= 0.80 → "High Risk"
RISK_THRESHOLD_MEDIUM = 0.60  # Prob_Fail >= 0.60 → "Medium Risk"
                               # Prob_Fail <  0.60 → "Low Risk" (if FAIL prediction)

def _risk(row):
    if row["Prediction"] == "PASS":                 return "Pass"
    elif row["Prob_Fail"] >= RISK_THRESHOLD_HIGH:   return "High Risk"
    elif row["Prob_Fail"] >= RISK_THRESHOLD_MEDIUM: return "Medium Risk"
    return "Low Risk"
```

Rows excluded (>30% missing):
- `Prediction = "N/A"`, `Risk_Level = "Excluded - Manual Review"`
- **Excluded from all statistics** in `get_stats()` — they are counted in `n_excluded`
  but NOT in `n_pass`, `n_fail`, `n_high`, `n_med`, `n_low`, `avg_c`

The `get_stats()` function (lines 303–334) returns:
```python
{"total", "n_pass", "n_fail", "n_high", "n_med", "n_low", "avg_c", "n_excluded"}
```

---

## 9. Known Issues and Active TODOs

### ISSUE-001 — `auditRecommendation` in `/api/executive-summary` is hardcoded
**File:** `backend/main.py`, line 609
```python
# Current (hardcoded):
"auditRecommendation": f"High risk alerts detected for {stats['n_high']} samples. Immediate quarantine recommended for flagged batches."
```
**Problem:** Does not reflect dominant failure category or active filter.
**Fix plan:**
1. Compute dominant failure category key from `f_pca` (as done for `topRisks`)
2. Map category → specific action:
   - `Microbiological` → "Quarantine and discard affected batches. Investigate hygiene controls."
   - `Physicochemical` → "Halt certification. Trace lipid oxidation in raw materials."
   - `Heavy_Metal` → "Mandatory supplier audit. Trace Pb/Cd contamination source."
   - `Stability` → "Review drying and packaging line humidity and temperature conditions."

---

### ISSUE-002 — Gemini fallback text is generic across all failure categories
**File:** `backend/main.py`, `generate_ai_narrative()`, lines 402–407
**Problem:** `next_steps` fallback always says "review [category] control points" —
no specific corrective action per failure type (e.g. no "discard batch" for Microbiological).
**Fix plan:**
Add `CATEGORY_NEXT_STEPS` dict:
```python
CATEGORY_NEXT_STEPS = {
    "Microbiological": "Quarantine and discard affected batches. Investigate hygiene controls and sanitization procedures before resubmission.",
    "Physicochemical": "Halt certification for affected batches. Trace raw material provenance for lipid oxidation sources.",
    "Heavy_Metal":     "Conduct mandatory supplier audit. Halt affected batches pending Pb/Cd source identification.",
    "Stability":       "Review drying and packaging line humidity conditions. Selective re-testing recommended for moisture-sensitive batches.",
}
```
Use as fallback when `top_category_label` maps to a known key.

---

### ISSUE-003 — `varExp` in `/api/clusters` is hardcoded
**File:** `backend/main.py`, line 579
```python
"varExp": {"pc1": 38.2, "pc2": 15.6}  # hardcoded
```
**Fix plan:** In `process_dataset()`, after `pca.fit_transform(df_feat.values)`, store:
```python
global global_pca_var_exp
global_pca_var_exp = {"pc1": round(pca.explained_variance_ratio_[0]*100, 1),
                       "pc2": round(pca.explained_variance_ratio_[1]*100, 1)}
```
Then use it in the `/api/clusters` endpoint.

---

### ISSUE-004 — `ExecutiveSummary.jsx` re-defines RISK_COLORS/BG/BORDER locally
**File:** `frontend/src/pages/ExecutiveSummary.jsx`, lines 16–18
```js
// Current (violation):
const RISK_COLORS = { High: '#E74C3C', Medium: '#F5B041', Low: '#2ECC71' }
const RISK_BG     = { High: '#FDEDEC', Medium: '#FEF9E7', Low: '#EAFAF1' }
const RISK_BORDER = { High: '#F5B7B1', Medium: '#F9E79F', Low: '#A9DFBF' }
```
**Fix plan:**
1. Remove lines 16–18
2. Add import: `import { RISK_COLORS, RISK_BG, RISK_BORDER } from '../constants/colors'`
3. Verify short key aliases (`High`, `Medium`, `Low`) exist in `colors.js` (they do)

---

## 10. Environment Variables

**File:** `.env` (project root, loaded by `python-dotenv` in `main.py` line 19)

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_AI_API_KEY` | Optional | Google Gemini API key. If absent or empty, `generate_ai_narrative()` falls back to template text silently (logs a warning). |

> **Security:** `.env` contains a live API key. Never commit to public repositories.
> Verify `.env` is listed in `.gitignore`.

---

## 11. Running the Project Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- Both runtime artifacts must exist:
  - `v4_model_audit/bqis_model_bundle.pkl`
  - `v2_feature_selection/bqis_clustering_result_v2.csv`

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate         # Windows PowerShell
# OR: source venv/bin/activate  # macOS/Linux

pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib fpdf2 python-dotenv google-genai

uvicorn main:app --reload --port 8000
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: `http://localhost:5173`

### Vite Proxy

`vite.config.js` proxies `/api/*` → `http://localhost:8000/api/*`.
Example: frontend calls `GET /api/dashboard` → backend receives `GET /api/dashboard`.

### Mock Mode (no backend required)

In `frontend/src/services/api.js`, set `const USE_MOCK = true`.
Returns static JSON from `frontend/src/mock/` with simulated 80ms delay.
Note: mock data does not reflect filter behavior.

---

## 12. ML Experiment History (Approaches 1–5)

| Approach | Status | Key Output (runtime?) | Key Finding |
|---|---|---|---|
| 1 — Baseline Clustering (11 features, no selection) | Archived | `bqis_clustering_result.csv` (not used) | K-Means ARI=0.130; DBSCAN ARI=0.281; both weak |
| 2 — Feature-Selected Clustering (MI top-5) | **In use** | `bqis_clustering_result_v2.csv` (**runtime dep**) | K-Means ARI=0.259, NMI=0.433; Stability purity=1.00 |
| 3 — Robust Pipeline (KNN impute, outlier flag, SNI hook) | Concluded | Pipeline in notebooks | 13.1% samples flagged as extreme outliers |
| 4 — Model Audit (leakage investigation, held-out test) | Concluded | `bqis_model_bundle.pkl` (**runtime dep**) | Partial leakage ~19pp; `Acid_Value_mgKOHg` corrected to 2.0 |
| 5 — Recalibration SOP (factory data onboarding) | Concluded | SOP notebook | 5-step protocol with mandatory human sign-off |

### Approach 2 — K-Means Cluster Purity

| Category | Purity |
|---|---|
| Stability | 1.00 — perfectly separated |
| Microbiological | 0.54 |
| Heavy_Metal | 0.51 |
| Physicochemical | 0.42 — lowest; inherently overlapping in all methods |

Overall: ARI=0.259, NMI=0.433 (K-Means, Mutual Information feature selection)

### Approach 4 — Partial Data Leakage Detail

| Test condition | CV Accuracy |
|---|---|
| With all features (including rule-based from dummy generator) | 0.967 ± 0.006 |
| Without 8 rule-based features | 0.775 |
| **Difference (leakage inflation)** | **~19 percentage points** |

**Conclusion:** Leakage is partial, not total. Model retains real signal (far above
~65% random baseline) but accuracy is inflated. This must be disclosed in all
presentations and demos.

### Approach 5 — Recalibration SOP (5 Stages)

1. **Data Intake and Mapping** — validate columns via `COLUMN_MAP`
2. **Quality Gate** — flag missing/outlier/duplicates; do NOT auto-remove outliers
3. **Mandatory Retraining** — model MUST be retrained from scratch on real data;
   MUST NOT inherit weights from dummy-data-trained `.pkl`
4. **Revalidation** — recompute feature selection, ARI, NMI from zero
5. **Human Sign-off and Model Card** — output JSON with `status: "PENDING_HUMAN_REVIEW"`;
   `approved_for_production` field must be set manually by authorized reviewer

---

## 13. SNI 2973:2022 Compliance Reference

| Parameter | SNI 2973:2022 Official Limit | Dataset Value | Status |
|---|---|---|---|
| Kadar air (Moisture_Content_%) | max 5.0% | 5.0% | Correct |
| Abu tidak larut asam (Acid_Insoluble_Ash_%) | max 0.1% | 0.1% | Correct |
| Bilangan asam (Acid_Value_mgKOHg) | max 2.0 mg KOH/g | Was 1.0 in dummy | **Corrected in Approach 4** |
| Timbal/Lead (Lead_Pb_mgkg) | max 0.50 mg/kg | 0.50 mg/kg | Correct |
| Kadmium/Cadmium (Cadmium_Cd_mgkg) | max 0.20 mg/kg | 0.20 mg/kg | Correct |
| Nilai peroksida (Peroxide_Value) | **NOT in SNI 2973:2022** | Used as ML feature | Not an official parameter |
| Timah/Tin (Sn) | max 40 mg/kg | Not in dataset | Out of scope for PoC |
| Merkuri/Mercury (Hg) | max 0.05 mg/kg | Not in dataset | Out of scope for PoC |
| Arsen/Arsenic (As) | max 0.50 mg/kg | Not in dataset | Out of scope for PoC |
| TPC, Kapang/Khamir (microbial limits) | Tiered per product type (SNI Tables 2–4) | Single generic limit | Known limitation — not product-specific |

---

## 14. Design Decisions and Constraints

### Why SHAP is global (not per-filter)
SHAP values represent the model's overall feature importance. Re-computing per filtered
subset would be expensive and unreliable on small subsets. Only counts/distributions are filtered.

### Why outliers are flagged but not removed
The BQIS Proposal states "high production batch variation" is a known limitation.
Automatic removal risks deleting real production variation. All outliers are flagged
in `Data_Quality_Flag` for human review only.

### Why clustering uses a pre-labeled CSV (not live K-Means)
Live K-Means would re-assign cluster IDs differently each run (random initialization).
The v2 CSV is fixed as ground truth for the PoC. For real factory data, the full
Approach 5 SOP must run before new cluster labels can be used.

### Why the model must be retrained for real factory data
The `.pkl` model was trained on synthetic data with known generation rules (partial leakage).
Approach 5 SOP mandates full retraining. The current bundle is for demo/PoC only.

### Why Peroxide_Value is in the model but not in SNI
Included in dummy dataset as a proxy for lipid oxidation. Approach 4 confirmed it is
not an official SNI 2973:2022 parameter. Retained for demo purposes; labeled
"outside SNI scope" in all methodology documentation.

### Color palette governance
Two files must always be kept **manually in sync**:

| File | Format |
|---|---|
| `frontend/src/constants/colors.js` | Hex strings (e.g. `#E74C3C`) |
| `backend/main.py` lines 717–723 | RGB tuples (e.g. `(231, 76, 60)`) |

**Palette A (authoritative):**
```
Microbiological: #E74C3C = rgb(231, 76, 60)
Physicochemical: #3498DB = rgb(52, 152, 219)
Heavy_Metal:     #9B59B6 = rgb(155, 89, 182)
Stability:       #F5B041 = rgb(245, 176, 65)
Pass:            #2ECC71 = rgb(46, 204, 113)
```
Do not introduce new category colors without updating both files simultaneously.
