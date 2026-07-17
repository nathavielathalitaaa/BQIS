# BQIS - Biscuit Quality Intelligence System

BQIS is a competition-ready research and decision-support platform for biscuit quality assessment under SNI 2973:2022. It combines machine learning, explainable AI, clustering, and automated reporting into a single workflow that turns laboratory data into clear audit actions.

This project is designed to be presented to judges, reviewers, and industry stakeholders. The focus is not only on prediction accuracy, but also on interpretability, operational usefulness, and a polished end-to-end experience.

## Why This Project Stands Out

BQIS is stronger than a simple classifier because it connects model output to real decision-making.

It provides:

- fast PASS / FAIL prediction from biscuit laboratory data
- risk grading that prioritizes samples for action
- explainable AI outputs using SHAP-style feature importance
- failure pattern grouping through PCA-based clustering
- technical and executive PDF reports for different audiences
- a clean React dashboard suitable for competition demos

For a competition or research presentation, this matters because it shows not just that the model works, but that the result is understandable, defensible, and usable in practice.

## Research And Competition Value

This project is a good fit for research showcases, innovation contests, and applied AI competitions because it demonstrates the full lifecycle of an intelligent quality system:

1. Problem framing around food quality certification.
2. Data preprocessing with missing-value handling.
3. Predictive modeling using XGBoost.
4. Explainability using SHAP contribution values.
5. Failure pattern discovery with PCA and pre-labeled clusters.
6. Risk-based decision support.
7. User-facing reporting for technical and non-technical audiences.

In short, BQIS is not only a model. It is a complete research prototype with a practical interface and a traceable methodology.

## Core Capabilities

BQIS can:

- classify each sample as PASS or FAIL
- assign High, Medium, Low, or Pass risk levels
- flag samples that need manual review when missing data is too high
- show which laboratory variables matter most globally
- map failure patterns into interpretable categories
- generate downloadable audit and executive reports
- support CSV upload and batch re-analysis

## System Architecture

The application is split into three layers:

1. Data layer: CSV datasets and experiment artifacts.
2. Backend layer: FastAPI service in backend/main.py.
3. Frontend layer: React + Vite dashboard in frontend/.

The backend loads the model bundle and dataset on startup, computes predictions and analytics in memory, and exposes JSON endpoints. The frontend consumes those endpoints to render charts, tables, summaries, and reports.

## Repository Structure

```text
BQIS/
|-- app.py                         # Legacy Streamlit prototype
|-- backend/
|   `-- main.py                    # FastAPI app, inference pipeline, report generation
|-- data/
|   |-- bqis_biscuit_quality_dataset.csv
|   `-- bqis_biscuit_quality_dataset_3000.csv
|-- frontend/
|   |-- src/
|   |   |-- App.jsx
|   |   |-- main.jsx
|   |   |-- index.css
|   |   |-- components/
|   |   |-- constants/
|   |   |-- mock/
|   |   |-- pages/
|   |   `-- services/
|   |-- package.json
|   `-- vite.config.js
|-- v1_baseline_full_features/
|-- v2_feature_selection/
|-- v3_robust_pipeline/
|-- v4_model_audit/
|-- v5_recalibration_sop/
`-- generate_dataset.py
```

Two runtime artifacts are required by the current backend:

- v4_model_audit/bqis_model_bundle.pkl
- v2_feature_selection/bqis_clustering_result_v2.csv

If either is missing, the backend cannot complete its analysis pipeline.

## Tech Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pandas and NumPy
- scikit-learn for KNN imputation and PCA
- XGBoost for binary classification
- fpdf2 for PDF generation
- python-dotenv for environment loading
- Google GenAI client for optional narrative text

### Frontend

- React 19
- Vite 8
- react-router-dom 7
- Axios
- Recharts
- Framer Motion
- react-icons

## Data Contract

The backend expects exact, case-sensitive column names in the CSV.

### Required Numeric Columns

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
| Product_Name | Product type used for one-hot encoding |

### Optional Columns

| Column | Default behavior if missing |
|---|---|
| Sample_ID | Auto-generated as UPL-0001, UPL-0002, and so on |
| Batch_Code | Filled with BCH-UPLOADED |
| Test_Date | Parsed if available for time filtering |

## Missing Data Policy

BQIS uses a clear quality gate for incomplete samples.

- Rows with 30% or less missing values are kept and imputed with KNNImputer.
- Rows with more than 30% missing values are excluded from model scoring and flagged for manual review.

This is useful in competition settings because it demonstrates a realistic industrial policy instead of silently forcing every record through the model.

## Backend Workflow

The backend entry point is backend/main.py.

### Startup Behavior

On startup, the FastAPI lifespan hook loads the baseline dataset from data/bqis_biscuit_quality_dataset.csv and processes it into cached analytics objects.

The main cached outputs are:

- global_ds: processed dataset with predictions and risk levels
- global_feat: feature matrix used for inference
- global_bundle: loaded model bundle and feature columns
- global_shap_df: global feature importance ranking
- global_pca_df: PCA coordinates with failure categories

### Processing Steps

The pipeline is intentionally easy to explain in a presentation:

1. Ensure required columns exist.
2. Fill missing identifiers if needed.
3. Measure row-level missingness.
4. Split included and excluded rows.
5. Impute included rows using KNN.
6. Encode Product_Name and align features.
7. Run XGBoost predictions.
8. Assign PASS / FAIL.
9. Convert model probability into a risk level.
10. Compute global feature importance.
11. Fit PCA for cluster visualization.
12. Merge pre-labeled failure categories.
13. Mark excluded rows for manual review.

### Risk Logic

The risk thresholds are:

- High Risk: failure probability at or above 0.80
- Medium Risk: failure probability at or above 0.60
- Low Risk: failure probability below the medium threshold
- Pass: any sample predicted as PASS

## API Endpoints

All endpoints live under /api.

### Filter Parameters

These endpoints accept optional filters:

- period: month-year string such as January 2025
- batch: exact Batch_Code value
- product: exact Product_Name value

### GET /api/dashboard

Returns the main competition-demo dashboard payload:

- total sample count
- pass and fail counts
- high-risk sample count
- average confidence
- pass and fail rates
- donut-chart distribution
- top SHAP parameters
- PCA scatter points

### GET /api/risk-overview

Returns:

- risk summary values
- risk table with action guidance
- the most recent samples

### GET /api/shap

Returns the global importance ranking with:

- parameter labels
- mean absolute SHAP values
- relative influence percentages
- feature direction

### GET /api/clusters

Returns the PCA cluster view and failure grouping summary.

### GET /api/executive-summary

Returns a business-level summary for management and competition judging.

### GET /api/filters/options

Returns available filter dropdown values derived from the dataset.

### POST /api/upload

Accepts a CSV file upload and reruns the full processing pipeline.

### GET /api/report/audit

Returns a technical PDF report for auditors and reviewers.

### GET /api/report/executive

Returns a concise executive PDF report for non-technical stakeholders.

## Frontend Experience

The frontend is built to look strong in a live demo.

It includes:

- a left navigation sidebar
- a dashboard with KPI cards and charts
- sample-level risk review
- failure-pattern visualization
- SHAP-based parameter importance
- an executive summary page

The app routes are defined in frontend/src/App.jsx:

- / - Dashboard
- /data-input - Data upload
- /sample-risk - Risk overview
- /failure-pattern - Failure pattern map
- /parameter-importance - Parameter importance
- /executive-summary - Executive summary

The frontend communicates with the backend through frontend/src/services/api.js, and Vite proxies /api to http://127.0.0.1:8000.

## Why The Visual Story Matters

For competition and research presentations, the visual layer is part of the scientific argument.

BQIS presents the model in a way that is easier to defend because it shows:

- what the model predicts
- why the model predicts it
- which samples need attention first
- how failures group into meaningful categories
- how the results are different for technical and executive audiences

That makes the system more persuasive than a black-box model alone.

## Local Setup

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variable

The backend optionally reads:

- GOOGLE_AI_API_KEY

If present, it can enrich the generated report narrative. If not present, the system still works using fallback text.

## Competition-Ready Strengths

- clear industrial use case
- complete end-to-end workflow
- explainable predictions, not just raw classification
- strong separation between technical and executive outputs
- polished dashboard demo value
- practical missing-data handling
- traceable experiment history in versioned folders

## Important Notes

- The backend is stateful in memory and reloads on restart.
- CSV column names must match exactly.
- The failure categories rely on the precomputed clustering CSV.
- SHAP values are global model behavior, not recalculated per filter.
- The project is intended for batch analytics and audit review, not real-time sensor control.

## Project Story In One Sentence

BQIS turns biscuit laboratory measurements into explainable quality intelligence that can be shown confidently in a research competition, technical audit, or executive review.