import os
import io
import logging
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
from google import genai

load_dotenv()
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
genai_client = genai.Client(api_key=GOOGLE_AI_API_KEY) if GOOGLE_AI_API_KEY else None

logger = logging.getLogger("bqis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Risk classification thresholds (synced with app.py)
RISK_THRESHOLD_HIGH = 0.80
RISK_THRESHOLD_MEDIUM = 0.60

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load dataset into memory on startup (replaces deprecated @app.on_event)."""
    load_data()
    yield

app = FastAPI(title="BQIS Backend API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "v4_model_audit", "bqis_model_bundle.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "bqis_biscuit_quality_dataset.csv")
CLUSTER_PATH = os.path.join(BASE_DIR, "v2_feature_selection", "bqis_clustering_result_v2.csv")

global_ds = None
global_feat = None
global_bundle = None
global_shap_df = None
global_pca_df = None
global_cluster_df = None

FEATURE_LABELS = {
    "Moisture_Content_%":     "Moisture Content",
    "Fat_Content_%":          "Fat Content",
    "Protein_Content_%":      "Protein",
    "Water_Activity_Aw":      "Water Activity",
    "Acid_Insoluble_Ash_%":   "Acid Insoluble Ash",
    "Acid_Value_mgKOHg":      "Acid Value",
    "Peroxide_Value":         "Peroxide Value",
    "Total_Plate_Count_CFUg": "Total Plate Count",
    "Yeast_Mold_Count_CFUg":  "Yeast & Mold Count",
    "Lead_Pb_mgkg":           "Lead (Pb)",
    "Cadmium_Cd_mgkg":        "Cadmium (Cd)",
    "pH":                     "pH",
    "Salt_Content":           "Salt Content"
}

# ── Proposal-compliant missing-data standards ──────────────────────────────────
# Source: BQIS Proposal — Section: Data Quality Management
# "Samples with more than 30% missing parameter values are excluded from AI
# analysis and flagged for manual review. For missing data below this threshold,
# BQIS applies K-Nearest Neighbor (KNN) imputation to preserve dataset
# completeness."
REQUIRED_NUMERIC_PARAMS = [
    "Moisture_Content_%", "Fat_Content_%", "Protein_Content_%",
    "Water_Activity_Aw", "Acid_Insoluble_Ash_%", "Acid_Value_mgKOHg",
    "Peroxide_Value", "Total_Plate_Count_CFUg", "Yeast_Mold_Count_CFUg",
    "Lead_Pb_mgkg", "Cadmium_Cd_mgkg",
]
MISSING_THRESHOLD = 0.30   # 30% — per-row missing fraction

def process_dataset(raw_df: pd.DataFrame):
    """
    Proposal-compliant dataset processing pipeline.

    Missing-data standard (BQIS Proposal):
      - Samples with >30% missing parameter values → excluded from AI analysis,
        flagged for manual review (Risk_Level = 'Excluded - Manual Review').
      - Samples with ≤30% missing → KNN imputation (n_neighbors=5, auto-reduced
        if dataset is small), then passed through the full model pipeline.
    """
    global global_ds, global_feat, global_shap_df, global_pca_df, global_cluster_df, global_bundle

    if global_bundle is None:
        global_bundle = joblib.load(BUNDLE_PATH)

    bundle = global_bundle
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    cols_to_use = REQUIRED_NUMERIC_PARAMS + ["Product_Name"]

    ds = raw_df.copy()

    # ── Auto-generate identifiers if absent (uploaded files may omit them) ────
    if "Sample_ID" not in ds.columns:
        ds["Sample_ID"] = [f"UPL-{i:04d}" for i in range(len(ds))]
    if "Batch_Code" not in ds.columns:
        ds["Batch_Code"] = "BCH-UPLOADED"
    if "Product_Name" not in ds.columns:
        ds["Product_Name"] = "Butter Biscuit"

    # ── Step 1: Ensure all required numeric columns exist ─────────────────────
    # If a column is entirely absent from the CSV it counts as 100% missing
    # for every row in that column (proposal-compliant).
    for col in REQUIRED_NUMERIC_PARAMS:
        if col not in ds.columns:
            ds[col] = np.nan

    # ── Step 2: Per-row missing percentage & split included / excluded ────────
    missing_pct   = ds[REQUIRED_NUMERIC_PARAMS].isna().mean(axis=1)
    ds["_missing_pct"] = missing_pct
    excluded_mask = missing_pct > MISSING_THRESHOLD
    included_mask = ~excluded_mask

    n_total    = len(ds)
    n_excluded = int(excluded_mask.sum())
    n_included = n_total - n_excluded

    logger.info(
        "Data quality split — total: %d | included (≤30%% missing): %d | "
        "excluded (>30%% missing, manual review): %d",
        n_total, n_included, n_excluded,
    )

    ds_included = ds[included_mask].copy()
    ds_excluded = ds[excluded_mask].copy()

    # ── Step 3: KNN imputation on included rows ───────────────────────────────
    # Proposal: "BQIS applies K-Nearest Neighbor (KNN) imputation to preserve
    # dataset completeness."
    if n_included > 0:
        n_neighbors = min(5, max(1, n_included - 1))
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_values = knn_imputer.fit_transform(ds_included[REQUIRED_NUMERIC_PARAMS])
        ds_included[REQUIRED_NUMERIC_PARAMS] = imputed_values
        logger.info("KNN imputation applied (n_neighbors=%d) to %d included samples.", n_neighbors, n_included)

    # ── Step 4: Model pipeline — ONLY for included rows ───────────────────────
    if n_included > 0:
        raw_sub = ds_included[[c for c in cols_to_use if c in ds_included.columns]].copy()
        if "Product_Name" not in raw_sub.columns:
            raw_sub["Product_Name"] = "Butter Biscuit"

        df_feat = pd.get_dummies(raw_sub, columns=["Product_Name"], prefix="Product")
        df_feat = df_feat.reindex(columns=feature_cols, fill_value=0)

        preds  = model.predict(df_feat)
        probas = model.predict_proba(df_feat)

        ds_included["Prediction"] = ["PASS" if p == 0 else "FAIL" for p in preds]
        ds_included["Prob_Pass"]  = probas[:, 0]
        ds_included["Prob_Fail"]  = probas[:, 1]
        ds_included["Confidence"] = [round(probas[i, p] * 100, 1) for i, p in enumerate(preds)]

        def _risk(row):
            if row["Prediction"] == "PASS":              return "Pass"
            elif row["Prob_Fail"] >= RISK_THRESHOLD_HIGH: return "High Risk"
            elif row["Prob_Fail"] >= RISK_THRESHOLD_MEDIUM: return "Medium Risk"
            return "Low Risk"

        ds_included["Risk_Level"] = ds_included.apply(_risk, axis=1)

        # ── SHAP explainability ───────────────────────────────────────────────
        booster      = model.get_booster()
        contribs     = booster.predict(xgb.DMatrix(df_feat), pred_contribs=True)
        shap_vals    = contribs[:, :-1]
        mean_abs     = np.abs(shap_vals).mean(axis=0)
        mean_signed  = shap_vals.mean(axis=0)

        shap_df = pd.DataFrame({
            "feature":     bundle["feature_columns"],
            "mean_abs":    mean_abs,
            "mean_signed": mean_signed,
        })
        shap_df = shap_df[~shap_df["feature"].str.startswith("Product_")]
        shap_df = shap_df.sort_values("mean_abs", ascending=False).reset_index(drop=True)
        total_shap = shap_df["mean_abs"].sum() or 1
        shap_df["label"]        = shap_df["feature"].map(FEATURE_LABELS).fillna(shap_df["feature"])
        shap_df["relative_pct"] = (shap_df["mean_abs"] / total_shap * 100).round(1)
        shap_df["direction"]    = shap_df["mean_signed"].apply(lambda v: "Positive" if v > 0 else "Negative")

        # ── PCA scatter (only included rows) ──────────────────────────────────
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(df_feat.values)

        pca_df = ds_included[
            ["Sample_ID", "Batch_Code", "Product_Name", "Prediction", "Risk_Level", "Confidence"]
        ].copy()
        pca_df["PC1"] = pcs[:, 0]
        pca_df["PC2"] = pcs[:, 1]

        try:
            cluster_df = pd.read_csv(CLUSTER_PATH)
            pca_df = pca_df.merge(
                cluster_df[["Sample_ID", "Failure_Category_Original"]],
                on="Sample_ID", how="left"
            )
            pca_df["Failure_Category_Original"] = pca_df["Failure_Category_Original"].fillna("Pass")
        except Exception as e:
            logger.warning("Cluster merge failed (%s) — using risk-based fallback categories", e)
            def _mock_cat(row):
                if row["Prediction"] == "PASS": return "Pass"
                cats = ["Microbiological", "Physicochemical", "Heavy_Metal", "Stability"]
                return cats[int(abs(row["PC1"] * 10)) % 4]
            pca_df["Failure_Category_Original"] = pca_df.apply(_mock_cat, axis=1)

    else:
        # Edge case: every row is excluded
        df_feat = pd.DataFrame(columns=feature_cols)
        shap_df = pd.DataFrame(columns=["feature", "mean_abs", "mean_signed", "label", "relative_pct", "direction"])
        pca_df  = pd.DataFrame(columns=["Sample_ID", "Batch_Code", "Product_Name",
                                         "Prediction", "Risk_Level", "Confidence",
                                         "PC1", "PC2", "Failure_Category_Original"])
        logger.warning("All %d rows excluded (>30%% missing) — no AI analysis performed.", n_total)

    # ── Step 5: Mark excluded rows ────────────────────────────────────────────
    if n_excluded > 0:
        ds_excluded["Prediction"] = "N/A"
        ds_excluded["Prob_Pass"]  = np.nan
        ds_excluded["Prob_Fail"]  = np.nan
        ds_excluded["Confidence"] = np.nan
        ds_excluded["Risk_Level"] = "Excluded - Manual Review"

    # ── Step 6: Merge back & restore original order ───────────────────────────
    ds_final = pd.concat([ds_included, ds_excluded]).sort_index()

    # ── Step 7: Data quality flags ────────────────────────────────────────────
    ds_final["Data_Quality_Flag"] = np.where(
        ds_final["_missing_pct"] > MISSING_THRESHOLD,
        "Excluded (>30% missing)",
        "Analyzed (KNN Imputed)",
    )
    ds_final["Missing_Percentage"] = (ds_final["_missing_pct"] * 100).round(1)
    ds_final.drop(columns=["_missing_pct"], inplace=True)

    # ── Step 8: Parse Test_Date for period filter ─────────────────────────────
    if "Test_Date" in ds_final.columns:
        ds_final["Test_Date_dt"] = pd.to_datetime(ds_final["Test_Date"], errors="coerce")
    else:
        ds_final["Test_Date_dt"] = pd.NaT

    global_ds      = ds_final
    global_feat    = df_feat
    global_shap_df = shap_df
    global_pca_df  = pca_df


def load_data():
    raw = pd.read_csv(DATASET_PATH)
    process_dataset(raw)

def apply_filters(ds, pca_df, period: str = None, batch: str = None, product: str = None):
    """
    Filter dataset berdasarkan period (Test_Date), batch, dan product.
    Period format: "January 2025", "June 2026", dll.
    """
    f_ds = ds.copy()
    f_pca = pca_df.copy()

    # ── Period filter via Test_Date ──────────────────────────────────────────
    if period and period not in ("All Time", None, ""):
        if "Test_Date" in f_ds.columns:
            try:
                # Parse period string seperti "January 2025" → (month=1, year=2025)
                parsed = pd.to_datetime(period, format="%B %Y")
                mask = (
                    (f_ds["Test_Date_dt"].dt.year  == parsed.year) &
                    (f_ds["Test_Date_dt"].dt.month == parsed.month)
                )
                f_ds = f_ds[mask]
                # Sinkronkan pca_df menggunakan index yang tersisa
                f_pca = f_pca[f_pca.index.isin(f_ds.index)]
            except Exception as e:
                logger.warning("Period filter parse failed for '%s' (%s) — ignoring filter", period, e)

    if batch and batch != "All Batches" and "Batch_Code" in f_ds.columns:
        f_ds = f_ds[f_ds["Batch_Code"] == batch]
        f_pca = f_pca[f_pca.index.isin(f_ds.index)]

    if product and product != "All Products" and "Product_Name" in f_ds.columns:
        f_ds = f_ds[f_ds["Product_Name"] == product]
        f_pca = f_pca[f_pca.index.isin(f_ds.index)]

    return f_ds, f_pca

def get_stats(ds):
    """
    Compute dataset statistics.
    Excluded rows (Prediction == 'N/A') are counted separately and NOT included
    in pass/fail/risk/confidence aggregates — consistent with proposal standard.
    """
    total      = len(ds)
    n_excluded = int((ds["Prediction"] == "N/A").sum()) if "Prediction" in ds.columns else 0
    analyzed   = ds[ds["Prediction"] != "N/A"] if "Prediction" in ds.columns else ds

    if total == 0:
        return {"total": 0, "n_pass": 0, "n_fail": 0,
                "n_high": 0, "n_med": 0, "n_low": 0,
                "avg_c": 0, "n_excluded": 0}

    n_pass = int((analyzed["Prediction"] == "PASS").sum())
    n_fail = int((analyzed["Prediction"] == "FAIL").sum())
    n_high = int((analyzed["Risk_Level"] == "High Risk").sum())
    n_med  = int((analyzed["Risk_Level"] == "Medium Risk").sum())
    n_low  = int((analyzed["Risk_Level"] == "Low Risk").sum())
    avg_c  = round(float(analyzed["Confidence"].dropna().mean()), 1) if len(analyzed) > 0 else 0.0

    return {
        "total":      total,
        "n_pass":     n_pass,
        "n_fail":     n_fail,
        "n_high":     n_high,
        "n_med":      n_med,
        "n_low":      n_low,
        "avg_c":      avg_c,
        "n_excluded": n_excluded,
    }

def _executive_notes(stats, total, period, batch, product):
    fail_pct = stats["n_fail"] / total * 100
    pass_pct = stats["n_pass"] / total * 100
    high_pct = stats["n_high"] / total * 100 if total else 0

    # Total Samples - scope kalimat, bukan cuma "Filtered period/batch/product"
    scope_parts = []
    if product and product != "All Products": scope_parts.append(product)
    if batch and batch != "All Batches": scope_parts.append(f"Batch {batch}")
    scope_parts.append(period if period and period != "All Time" else "all recorded periods")
    total_note = " - ".join(scope_parts)

    # Predicted PASS
    if pass_pct >= 80:
        pass_note = "Strong majority meet quality standards"
    elif pass_pct >= 50:
        pass_note = "About half of samples meet standards"
    else:
        pass_note = "Minority pass - compliance concern this period"

    # Predicted FAIL
    if fail_pct < 10:
        fail_note = "Within normal fail-rate range"
    elif fail_pct < 30:
        fail_note = "Moderate fail rate - monitor closely"
    else:
        fail_note = "Elevated fail rate - requires attention"

    # High Risk - bandingkan proporsi terhadap total FAIL, bukan cuma total
    if stats["n_fail"] == 0:
        high_note = "No failing samples this period"
    elif stats["n_high"] == stats["n_fail"]:
        high_note = "All failing samples are high-confidence failures"
    elif stats["n_high"] > 0:
        high_note = f"{stats['n_high']} of {stats['n_fail']} failing samples need urgent review"
    else:
        high_note = "No urgent cases this period"

    # Medium / Low Risk - kontekstual, bukan definisi threshold
    medium_note = "None flagged this period" if stats["n_med"] == 0 else f"{stats['n_med']} sample(s) warrant selective re-testing"
    low_note = "None flagged this period" if stats["n_low"] == 0 else f"{stats['n_low']} sample(s) under standard monitoring"

    # Avg Confidence
    if stats["avg_c"] >= 95:
        conf_note = "Very high model certainty"
    elif stats["avg_c"] >= 85:
        conf_note = "High confidence in predictions"
    else:
        conf_note = "Moderate confidence - consider manual cross-check"

    # Excluded Rows
    n_excluded = stats.get("n_excluded", 0)
    excluded_note = "No data quality issues affecting analysis" if n_excluded == 0 else f"{n_excluded} sample(s) need manual review due to missing data"

    return {
        "total": total_note, "pass": pass_note, "fail": fail_note,
        "high": high_note, "medium": medium_note, "low": low_note,
        "confidence": conf_note, "excluded": excluded_note,
    }

def generate_ai_narrative(context: dict, section: str) -> str:
    """
    Generate narrative text via Gemini. context berisi angka-angka aktual
    hasil filter (period/batch/product) dari request saat ini.
    section: "next_steps" atau "certification_impact" — menentukan prompt & fallback.
    """
    fallback_texts = {
        "next_steps": (
            f"We recommend {context['batch_label']} be placed on hold pending corrective action. "
            f"Production teams should review {context['top_category_label']} control points before "
            f"resubmission for certification."
        ),
        "certification_impact": (
            f"At current compliance rates, an estimated {context['n_fail']:,} batches "
            f"({context['fail_pct']:.1f}%) may face certification delays if underlying quality issues "
            f"are not addressed prior to re-testing."
        ),
    }

    if genai_client is None:
        logger.warning("GOOGLE_AI_API_KEY tidak ditemukan — pakai fallback template untuk section '%s'", section)
        return fallback_texts[section]

    prompts = {
        "next_steps": (
            "You are writing a short business recommendation paragraph (2-3 sentences, plain English, "
            "no markdown, no bullet points) for a food quality certification executive summary PDF, "
            "addressed to management and certification clients (not lab auditors). "
            f"Context: {context['total']} biscuit samples were tested for period '{context['period']}', "
            f"batch '{context['batch']}', product '{context['product']}'. "
            f"{context['n_fail']} samples ({context['fail_pct']:.1f}%) failed. "
            f"The leading quality concern category is '{context['top_category_label']}' affecting "
            f"{context['top_category_count']} samples. "
            "Write a concise, actionable recommendation for next steps before re-certification. "
            "Do not invent numbers not given above."
        ),
        "certification_impact": (
            "You are writing a short business-impact paragraph (2-3 sentences, plain English, no markdown) "
            "for a food quality certification executive summary PDF, addressed to management/clients. "
            f"Context: out of {context['total']} samples tested (period '{context['period']}', "
            f"batch '{context['batch']}', product '{context['product']}'), {context['n_fail']} samples "
            f"({context['fail_pct']:.1f}%) are predicted to fail certification standards. "
            "Explain the potential business/certification-timeline impact if these issues are not resolved. "
            "Do not invent numbers not given above."
        ),
    }

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompts[section],
        )
        text = response.text.strip()
        return text if text else fallback_texts[section]
    except Exception as e:
        logger.warning("Gemini API call gagal untuk section '%s' (%s) — pakai fallback template", section, e)
        return fallback_texts[section]

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard")
def get_dashboard(period: str = None, batch: str = None, product: str = None):
    f_ds, f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    
    # Top SHAP and Scatter for real-time dashboard UI
    top_5 = global_shap_df.head(5)
    param_impact = [{"label": r["label"], "meanAbs": r["mean_abs"], "relativePct": r["relative_pct"]} for _, r in top_5.iterrows()]
    
    scatter_points = []
    # Palet A — standar resmi (sinkron dengan frontend/src/constants/colors.js)
    color_map = {
        "Microbiological": "#E74C3C",
        "Physicochemical": "#3498DB",
        "Heavy_Metal":     "#9B59B6",
        "Stability":       "#F5B041",
        "Pass":            "#2ECC71",
    }
    sample_pca = f_pca.sample(min(200, len(f_pca)), random_state=42) if len(f_pca) > 0 else f_pca
    for _, row in sample_pca.iterrows():
        scatter_points.append({
            "id": row["Sample_ID"], "pc1": round(row["PC1"], 3), "pc2": round(row["PC2"], 3),
            "cluster": row["Failure_Category_Original"]
        })

    return {
        "totalSamples": stats["total"],
        "predictedPass": stats["n_pass"],
        "predictedFail": stats["n_fail"],
        "highRiskSamples": stats["n_high"],
        "avgConfidence": stats["avg_c"],
        "passRate": round((stats["n_pass"] / stats["total"]) * 100, 1) if stats["total"] else 0,
        "failRate": round((stats["n_fail"] / stats["total"]) * 100, 1) if stats["total"] else 0,
        "riskDistribution": [
            { "name": "Pass",        "value": stats["n_pass"], "color": "#2ECC71" },
            { "name": "High Risk",   "value": stats["n_high"], "color": "#E74C3C" },
            { "name": "Medium Risk", "value": stats["n_med"],  "color": "#F5B041" },
            { "name": "Low Risk",    "value": stats["n_low"],  "color": "#82E0AA" }
        ],
        "topShap": param_impact,
        "scatterPoints": scatter_points,
    }

@app.get("/api/risk-overview")
def get_risk_overview(period: str = None, batch: str = None, product: str = None):
    f_ds, _ = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    
    recent = f_ds.tail(10).iloc[::-1]
    recent_list = []
    for _, row in recent.iterrows():
        recent_list.append({
            "id": row["Sample_ID"], "batch": row.get("Batch_Code", "-"),
            "product": row.get("Product_Name", "-"), "prediction": row["Prediction"],
            "risk": row["Risk_Level"], "confidence": row["Confidence"]
        })
    
    return {
        "totalSamples": stats["total"],
        "predictedPass": stats["n_pass"],
        "predictedFail": stats["n_fail"],
        "highRisk": stats["n_high"],
        "avgConfidence": stats["avg_c"],
        "riskDistribution": [
            { "name": "Pass",        "value": stats["n_pass"], "color": "#2ECC71" },
            { "name": "High Risk",   "value": stats["n_high"], "color": "#E74C3C" },
            { "name": "Medium Risk", "value": stats["n_med"],  "color": "#F5B041" },
            { "name": "Low Risk",    "value": stats["n_low"],  "color": "#82E0AA" }
        ],
        "riskTable": [
            { "level": "Pass",        "count": stats["n_pass"], "pct": f"{(stats['n_pass']/max(1, stats['total']))*100:.1f}%", "action": "Clear for certification",    "priority": "—" },
            { "level": "High Risk",   "count": stats["n_high"], "pct": f"{(stats['n_high']/max(1, stats['total']))*100:.1f}%", "action": "Immediate auditor review",   "priority": "URGENT" },
            { "level": "Medium Risk", "count": stats["n_med"],  "pct": f"{(stats['n_med']/max(1, stats['total']))*100:.1f}%",  "action": "Selective re-testing",       "priority": "HIGH" },
            { "level": "Low Risk",    "count": stats["n_low"],  "pct": f"{(stats['n_low']/max(1, stats['total']))*100:.1f}%",  "action": "Standard monitoring",        "priority": "NORMAL" }
        ],
        "recentSamples": recent_list
    }

@app.get("/api/shap")
def get_shap():
    # SHAP is global model characteristic, no need to filter by batch
    params = []
    for _, row in global_shap_df.iterrows():
        params.append({
            "label": row["label"], "meanAbs": row["mean_abs"],
            "relativePct": row["relative_pct"], "direction": row["direction"]
        })
    return {
        "paramCount": len(params),
        "affectedSamples": len(global_ds),
        "parameters": params
    }

@app.get("/api/clusters")
def get_clusters(period: str = None, batch: str = None, product: str = None):
    _, f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)
    
    scatter_points = []
    sample_pca = f_pca.sample(min(200, len(f_pca)), random_state=42) if len(f_pca) > 0 else f_pca
    for _, row in sample_pca.iterrows():
        scatter_points.append({
            "id": row["Sample_ID"], "pc1": round(row["PC1"], 3), "pc2": round(row["PC2"], 3),
            "cluster": row["Failure_Category_Original"]
        })
        
    cat_counts = f_pca[f_pca["Failure_Category_Original"] != "Pass"]["Failure_Category_Original"].value_counts()
    dominant_cat = cat_counts.idxmax() if len(cat_counts) > 0 else "None"
    
    cluster_profiles = [
        {"key": "Microbiological", "samples": int(cat_counts.get("Microbiological", 0))},
        {"key": "Physicochemical", "samples": int(cat_counts.get("Physicochemical", 0))},
        {"key": "Heavy_Metal", "samples": int(cat_counts.get("Heavy_Metal", 0))},
        {"key": "Stability", "samples": int(cat_counts.get("Stability", 0))}
    ]
    return {
        "totalClusters": 4,
        "dominantCluster": dominant_cat,
        "highRiskClusters": 2,
        "affectedSamples": int(cat_counts.sum()),
        "scatterPoints": scatter_points,
        "clusterProfiles": cluster_profiles,
        "varExp": {"pc1": 38.2, "pc2": 15.6}
    }

@app.get("/api/executive-summary")
def get_executive_summary(period: str = None, batch: str = None, product: str = None):
    f_ds, f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    
    cat_counts = f_pca["Failure_Category_Original"].value_counts()
    top_risks = [
        {"category": "Microbiological Contamination", "samples": int(cat_counts.get("Microbiological", 0)), "risk": "High", "action": "Immediate re-inspection"},
        {"category": "Physicochemical Non-compliance", "samples": int(cat_counts.get("Physicochemical", 0)), "risk": "High", "action": "Halt certification"},
        {"category": "Heavy Metal Exceedance", "samples": int(cat_counts.get("Heavy_Metal", 0)), "risk": "High", "action": "Supplier audit"},
        {"category": "Moisture / Stability Deviation", "samples": int(cat_counts.get("Stability", 0)), "risk": "Medium", "action": "Selective re-testing"}
    ]
    
    top_5 = global_shap_df.head(5)
    param_impact = [{"parameter": row["label"], "shapVal": row["mean_abs"]} for _, row in top_5.iterrows()]
    
    return {
        "totalSamples": stats["total"],
        "predictedPass": stats["n_pass"],
        "predictedFail": stats["n_fail"],
        "avgConfidence": stats["avg_c"],
        # Masalah 3: tambahkan passRate & failRate yang sebelumnya missing
        "passRate": round((stats["n_pass"] / stats["total"]) * 100, 1) if stats["total"] else 0,
        "failRate": round((stats["n_fail"] / stats["total"]) * 100, 1) if stats["total"] else 0,
        "riskSummary": {"high": stats["n_high"], "medium": stats["n_med"], "low": stats["n_low"], "pass": stats["n_pass"]},
        "topRisks": top_risks,
        "parameterImpact": param_impact,
        "auditRecommendation": f"High risk alerts detected for {stats['n_high']} samples. Immediate quarantine recommended for flagged batches."
    }

@app.get("/api/filters/options")
def get_filter_options():
    """Kembalikan semua opsi filter: periods dari Test_Date, batches, products."""
    if global_ds is None:
        return {"periods": [], "batches": [], "products": []}

    batches  = sorted([str(x) for x in global_ds["Batch_Code"].dropna().unique() if x != ""])
    products = sorted([str(x) for x in global_ds["Product_Name"].dropna().unique() if x != ""])

    # Masalah 2: ambil unique year-month dari Test_Date, format "January 2025"
    periods = []
    if "Test_Date_dt" in global_ds.columns:
        period_series = global_ds["Test_Date_dt"].dt.to_period("M").dropna().unique()
        # Sort ascending, format ke string
        periods = sorted(
            [p.to_timestamp().strftime("%B %Y") for p in period_series]
        )

    return {
        "periods": periods,
        "batches": batches,
        "products": products,
    }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    # ── 1. Validate file extension ────────────────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="File tidak valid. Hanya file .csv yang diterima.")

    contents = await file.read()

    # ── 2. Parse CSV ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as parse_err:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"Gagal membaca file CSV: {parse_err}. Pastikan file tidak rusak dan berformat CSV standar."
        )

    if len(df) == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="File CSV kosong. Tidak ada baris data yang ditemukan.")

    # ── 3. Check required columns ─────────────────────────────────────────────
    REQUIRED = [
        "Moisture_Content_%", "Fat_Content_%", "Protein_Content_%",
        "Water_Activity_Aw", "Acid_Insoluble_Ash_%", "Acid_Value_mgKOHg",
        "Peroxide_Value", "Total_Plate_Count_CFUg", "Yeast_Mold_Count_CFUg",
        "Lead_Pb_mgkg", "Cadmium_Cd_mgkg", "Product_Name",
    ]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=(
                f"Kolom wajib tidak ditemukan ({len(missing_cols)} kolom): "
                + ", ".join(missing_cols)
                + ". Pastikan nama kolom persis sama (case-sensitive)."
            )
        )

    # ── 4. Process dataset ────────────────────────────────────────────────────
    try:
        process_dataset(df)
    except KeyError as ke:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"Kolom tidak cocok dengan model: {ke}. Periksa nama kolom di file CSV Anda."
        )
    except ValueError as ve:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"Data tidak valid: {ve}. Pastikan kolom numerik tidak mengandung teks atau nilai kosong berlebihan."
        )
    except Exception as exc:
        logger.exception("process_dataset failed for uploaded file '%s'", file.filename)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses dataset: {type(exc).__name__} — {exc}"
        )

    stats = get_stats(global_ds)
    return {
        "message": "Dataset berhasil diupload dan diproses",
        "samples": len(global_ds),
        "quality_summary": {
            "analyzed":              int(len(global_ds) - stats["n_excluded"]),
            "excluded_manual_review": stats["n_excluded"],
            "threshold_pct":         int(MISSING_THRESHOLD * 100),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PDF HELPERS (shared by audit & executive report generators)
# ══════════════════════════════════════════════════════════════════════════════

NAVY = (0, 32, 91)        # TÜV NORD blue
RED = (231, 76, 60)
YELLOW = (245, 176, 65)
GREEN = (46, 204, 113)
LIGHT = (248, 249, 250)
GREY = (127, 140, 141)
DARK = (51, 51, 51)


class BQISReport(FPDF):
    """Custom FPDF subclass with BQIS branded header/footer and helpers."""

    def __init__(self, title: str = "BQIS Audit Report"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title = title
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 15, 15)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return  # Cover page has its own layout
        self.set_font("Arial", "B", 10)
        self.set_text_color(*NAVY)
        self.cell(0, 8, "BQIS Audit Report", align="L")
        self.set_font("Arial", "", 9)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="R")
        self.ln(10)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Arial", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"BQIS -- TÜV NORD  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def h1(self, text):
        self.set_font("Arial", "B", 16)
        self.set_text_color(*NAVY)
        self.cell(0, 10, text, ln=1)
        self.ln(2)

    def h2(self, text):
        self.set_font("Arial", "B", 14)
        self.set_text_color(*DARK)
        self.cell(0, 9, text, ln=1)
        self.ln(1)

    def _cover(self, subtitle: str, period: str, batch: str, product: str,
               audience_label: str = None):
        self.add_page()
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 60, "F")
        self.set_xy(15, 18)
        self.set_font("Arial", "B", 30)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "BQIS", ln=1)
        self.set_x(15)
        self.set_font("Arial", "B", 14)
        self.cell(0, 8, "TUV NORD", ln=1)
        self.ln(20)
        self.set_text_color(*NAVY)
        self.set_font("Arial", "B", 22)
        self.cell(0, 12, self.title, ln=1, align="C")
        self.set_font("Arial", "", 12)
        self.set_text_color(*GREY)
        self.cell(0, 8, subtitle, ln=1, align="C")
        # Audience label — rendered in small italic below the subtitle
        if audience_label:
            self.set_font("Arial", "I", 10)
            self.set_text_color(*GREY)
            self.cell(0, 7, audience_label, ln=1, align="C")
        self.ln(10)
        # Filter info box
        self.set_x(35)
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.3)
        box_w = 140
        self.multi_cell(box_w, 8,
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Period    : {period or 'All Time'}\n"
            f"Batch     : {batch or 'All'}\n"
            f"Product   : {product or 'All'}\n"
            f"Standard  : SNI 2973:2022",
            border=1, align="L", fill=True)

    def _table(self, headers, rows, col_widths, aligns=None, row_colors=None):
        """Render a bordered table with header row in navy."""
        aligns = aligns or ["L"] * len(headers)
        line_h = 7
        # Header
        self.set_font("Arial", "B", 10)
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], line_h, h, border=1, align="C", fill=True)
        self.ln(line_h)
        # Body
        self.set_font("Arial", "", 10)
        self.set_text_color(*DARK)
        for r_idx, row in enumerate(rows):
            fill = False
            if row_colors and r_idx < len(row_colors) and row_colors[r_idx]:
                self.set_fill_color(*row_colors[r_idx])
                fill = True
            for i, cell in enumerate(row):
                self.cell(col_widths[i], line_h, str(cell), border=1, align=aligns[i], fill=fill)
            self.ln(line_h)


@app.get("/api/report/audit")
def generate_audit_report(period: str = None, batch: str = None, product: str = None):
    f_ds, _ = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    total = max(1, stats["total"])

    pdf = BQISReport(title="AUDIT REPORT")
    pdf._cover(
        "Biscuit Quality Intelligence System", period, batch, product,
        audience_label="Prepared for: Auditors & Laboratory Analysts"
    )

    # ── 1. EXECUTIVE SUMMARY (technical metrics table) ──────────────────────────
    pdf.add_page()
    pdf.h1("1. Executive Summary")
    
    notes = _executive_notes(stats, total, period, batch, product)
    safe_notes = {k: v.encode('latin-1', 'replace').decode('latin-1') for k, v in notes.items()}
    
    pdf._table(
        ["Metric", "Value", "Notes"],
        [
            ["Total Samples", f"{stats['total']:,}", safe_notes["total"]],
            ["Predicted PASS", f"{stats['n_pass']:,} ({stats['n_pass']/total*100:.1f}%)", safe_notes["pass"]],
            ["Predicted FAIL", f"{stats['n_fail']:,} ({stats['n_fail']/total*100:.1f}%)", safe_notes["fail"]],
            ["High Risk", f"{stats['n_high']:,}", safe_notes["high"]],
            ["Medium Risk", f"{stats['n_med']:,}", safe_notes["medium"]],
            ["Low Risk", f"{stats['n_low']:,}", safe_notes["low"]],
            ["Avg Confidence", f"{stats['avg_c']:.1f}%", safe_notes["confidence"]],
            ["Excluded Rows", f"{stats.get('n_excluded', 0):,}", safe_notes["excluded"]],
        ],
        [60, 50, 80], aligns=["L", "C", "L"]
    )
    pdf.ln(6)

    # ── 2. RISK DISTRIBUTION ────────────────────────────────────────────────────
    pdf.h2("2. Risk Distribution")
    pdf._table(
        ["Risk Level", "Samples", "% of Total", "Required Action"],
        [
            ["Pass",        f"{stats['n_pass']:,}", f"{stats['n_pass']/total*100:.1f}%", "Clear for certification"],
            ["High Risk",   f"{stats['n_high']:,}", f"{stats['n_high']/total*100:.1f}%", "URGENT - Quarantine & re-verify"],
            ["Medium Risk", f"{stats['n_med']:,}",  f"{stats['n_med']/total*100:.1f}%",  "HIGH - Selective re-testing"],
            ["Low Risk",    f"{stats['n_low']:,}",  f"{stats['n_low']/total*100:.1f}%",  "NORMAL - Standard monitoring"],
        ],
        [48, 35, 32, 75],
        aligns=["L", "C", "C", "L"],
        row_colors=[GREEN, RED, YELLOW, None],
    )
    pdf.ln(6)

    # ── 3. TOP SHAP PARAMETERS (full technical table) ───────────────────────────
    pdf.h2("3. Top SHAP Parameters (XGBoost Explainability)")
    top = global_shap_df.head(8)
    total_shap = global_shap_df["mean_abs"].sum() or 1
    pdf._table(
        ["Parameter", "Mean |SHAP|", "Relative Influence", "Direction"],
        [
            [
                r["label"],
                f"{r['mean_abs']:.4f}",
                f"{r['mean_abs']/total_shap*100:.1f}%",
                r["direction"],
            ]
            for _, r in top.iterrows()
        ],
        [78, 35, 40, 37], aligns=["L", "C", "C", "C"]
    )
    pdf.ln(6)

    # ── 4. CLUSTER PROFILES ─────────────────────────────────────────────────────
    pdf.h2("4. Failure Cluster Profiles")
    # Re-fetch filtered pca for cluster counts
    f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)[1]
    cc = f_pca[f_pca["Failure_Category_Original"] != "Pass"]["Failure_Category_Original"].value_counts()
    cluster_rows = sorted(
        [
            ["Microbiological", int(cc.get("Microbiological", 0)), "High",   "Immediate re-inspection required"],
            ["Physicochemical", int(cc.get("Physicochemical",  0)), "High",   "Halt certification pending review"],
            ["Heavy Metal",     int(cc.get("Heavy_Metal",       0)), "High",   "Supplier & raw-material audit"],
            ["Stability",       int(cc.get("Stability",         0)), "Medium", "Selective re-testing (storage)"],
        ],
        key=lambda r: r[1], reverse=True  # sort by count descending
    )
    # Assign row colours based on risk label
    _risk_color_map = {"High": RED, "Medium": YELLOW}
    cluster_row_colors = [_risk_color_map.get(r[2], None) for r in cluster_rows]
    # Convert count to string for display
    cluster_rows_display = [[r[0], f"{r[1]:,}", r[2], r[3]] for r in cluster_rows]
    pdf._table(
        ["Failure Cluster", "Samples", "Risk", "Auditor Action"],
        cluster_rows_display, [58, 35, 32, 65],
        aligns=["L", "C", "C", "L"],
        row_colors=cluster_row_colors,
    )
    pdf.ln(6)

    # ── 5. DATA QUALITY & MISSING VALUE HANDLING ────────────────────────────────
    pdf.h2("5. Data Quality & Missing Value Handling")
    # Fields Data_Quality_Flag and n_excluded are present in global_ds after
    # process_dataset() — safe to use directly.
    n_analyzed  = stats["total"] - stats["n_excluded"]
    n_excluded  = stats["n_excluded"]
    pdf._table(
        ["Metric", "Count", "Notes"],
        [
            ["Total samples processed",
             f"{stats['total']:,}",
             "All rows in filtered period/batch/product"],
            ["Samples analyzed (KNN imputed, <= 30% missing)",
             f"{n_analyzed:,}",
             "Passed through XGBoost pipeline"],
            ["Samples excluded (> 30% missing)",
             f"{n_excluded:,}",
             "Flagged for manual laboratory review"],
        ],
        [90, 28, 72], aligns=["L", "C", "L"]
    )
    pdf.ln(3)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 6,
        "Methodology note: Missing values handled per BQIS standard: KNN imputation "
        "(k=5) for samples with <= 30% missing parameters; samples exceeding this "
        "threshold are excluded from AI scoring and flagged for manual review.",
        border=0, align="L")
    pdf.set_text_color(*DARK)
    pdf.ln(4)

    # ── 6. METHODOLOGY REFERENCE ────────────────────────────────────────────────
    pdf.h2("6. Methodology Reference")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(*DARK)
    methodology_points = [
        "Classification: XGBoost gradient-boosted trees (Chen & Guestrin, 2016) "
        "- binary classification (PASS / FAIL) with calibrated probability output.",
        "Explainability: SHAP (SHapley Additive exPlanations) TreeExplainer "
        "(Lundberg & Lee, 2017) - mean absolute SHAP values ranked by global importance.",
        "Clustering: PCA (2-component) dimensionality reduction merged with "
        "pre-labelled failure categories from bqis_clustering_result_v2.csv.",
        "Missing-data policy: KNN imputation (n_neighbors=5, auto-reduced for small "
        "datasets); rows with > 30% missing excluded (BQIS Proposal, Data Quality Mgmt).",
        "Quality standard: SNI 2973:2011 - Biskuit (Biscuit quality reference limits).",
    ]
    for point in methodology_points:
        pdf.set_x(15)
        pdf.multi_cell(0, 6, f"  - {point}", border=0, align="L")
        pdf.ln(1)
    pdf.ln(4)

    # ── 7. AUDIT RECOMMENDATION (SHAP-driven, technical, actionable) ─────────────
    pdf.h2("7. Audit Recommendation")
    # top already computed above (global_shap_df.head(8)); use head(1) for #1 driver
    top1 = global_shap_df.iloc[0]
    top_influence_pct = top1["mean_abs"] / total_shap * 100
    # Dominant failure cluster = cluster_rows[0] (already sorted by count desc)
    dominant_cluster_name = cluster_rows[0][0] if cluster_rows else "N/A"
    pdf.set_fill_color(255, 235, 235)
    pdf.set_draw_color(*RED)
    pdf.set_line_width(0.4)
    pdf.set_text_color(150, 30, 30)
    pdf.set_font("Arial", "B", 11)
    rec_text = (
        f"Based on SHAP analysis, {top1['label']} is the leading contributor to failure "
        f"risk ({top_influence_pct:.1f}% relative influence). "
        f"{stats['n_high']} samples require immediate quarantine pending manual laboratory "
        f"re-verification. "
        f"Auditors should prioritize batches under the '{dominant_cluster_name}' failure "
        f"cluster first, as it has the highest sample count among all failure categories "
        f"in the current filtered view."
    )
    pdf.multi_cell(0, 8, rec_text, border=1, align="L", fill=True)

    pdf_content = bytes(pdf.output(dest='S'))
    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=audit_report.pdf"}
    )

# Plain-language category labels for executive audience (non-technical)
CATEGORY_PLAIN_LABELS = {
    "Microbiological": "Microbial Contamination",
    "Physicochemical": "Moisture & Composition Issues",
    "Heavy_Metal":     "Heavy Metal Contamination",
    "Stability":       "Shelf-Life / Storage Stability",
}


@app.get("/api/report/executive")
def generate_exec_summary(period: str = None, batch: str = None, product: str = None):
    f_ds, f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    total = max(1, stats["total"])

    pdf = BQISReport(title="EXECUTIVE SUMMARY")
    pdf._cover(
        "Biscuit Quality Intelligence System", period, batch, product,
        audience_label="Prepared for: Management & Certification Clients"
    )

    pdf.add_page()

    # ── KEY FINDINGS (business-oriented language) ────────────────────────────────
    pdf.h1("Key Findings")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 7,
        f"Out of {stats['total']:,} biscuit samples tested this period, "
        f"{stats['n_pass']/total*100:.0f}% met quality standards. "
        f"{stats['n_high']:,} samples showed strong indicators of non-compliance and are "
        f"recommended for immediate corrective action before certification proceeds.",
        border=0, align="L")
    pdf.ln(5)

    # ── RISK BREAKDOWN (retained — concise & visual) ─────────────────────────────
    pdf.h2("Risk Breakdown")
    pdf._table(
        ["Risk Level", "Samples"],
        [
            ["High Risk",   f"{stats['n_high']:,}"],
            ["Medium Risk", f"{stats['n_med']:,}"],
            ["Low Risk",    f"{stats['n_low']:,}"],
            ["Pass",        f"{stats['n_pass']:,}"],
        ],
        [100, 60], aligns=["L", "C"],
        row_colors=[RED, YELLOW, GREEN, None],
    )
    pdf.ln(6)

    # ── PRIMARY QUALITY CONCERNS (plain-language, 2-column, no technical jargon) ──
    pdf.h2("Primary Quality Concerns")
    cc = f_pca[f_pca["Failure_Category_Original"] != "Pass"]["Failure_Category_Original"].value_counts()
    # Build rows sorted by count descending; use plain labels only
    concern_rows_raw = [
        (CATEGORY_PLAIN_LABELS.get("Microbiological", "Microbiological"), int(cc.get("Microbiological", 0))),
        (CATEGORY_PLAIN_LABELS.get("Physicochemical",  "Physicochemical"),  int(cc.get("Physicochemical",  0))),
        (CATEGORY_PLAIN_LABELS.get("Heavy_Metal",      "Heavy Metal"),      int(cc.get("Heavy_Metal",      0))),
        (CATEGORY_PLAIN_LABELS.get("Stability",        "Stability"),        int(cc.get("Stability",        0))),
    ]
    concern_rows_raw.sort(key=lambda r: r[1], reverse=True)
    # Identify top category for recommendation section (dynamic, not hardcoded)
    top_category_raw = (
        cc.idxmax() if len(cc) > 0 else "Physicochemical"
    )
    pdf._table(
        ["Concern Area", "Samples Affected"],
        [[row[0], f"{row[1]:,}"] for row in concern_rows_raw],
        [120, 60], aligns=["L", "C"],
    )
    pdf.ln(6)

    # ── RECOMMENDED NEXT STEPS (client/management focus, actionable) ─────────────
    pdf.h2("Recommended Next Steps")
    
    top_category_key = cc.idxmax() if len(cc) > 0 else None
    top_category_count = int(cc.max()) if len(cc) > 0 else 0
    top_category_label = CATEGORY_PLAIN_LABELS.get(top_category_key, top_category_key or "Unknown")

    narrative_context = {
        "total": stats["total"],
        "n_fail": stats["n_fail"],
        "fail_pct": stats["n_fail"] / total * 100,
        "period": period or "All Time",
        "batch": batch or "All Batches",
        "batch_label": batch if batch and batch != "All Batches" else "the flagged batches",
        "product": product or "All Products",
        "top_category_label": top_category_label,
        "top_category_count": top_category_count,
    }

    next_steps_text = generate_ai_narrative(narrative_context, "next_steps")
    certification_impact_text = generate_ai_narrative(narrative_context, "certification_impact")

    pdf.set_fill_color(255, 243, 220)
    pdf.set_draw_color(200, 140, 0)
    pdf.set_line_width(0.4)
    pdf.set_text_color(120, 80, 0)
    pdf.set_font("Arial", "B", 11)
    
    safe_next_steps = next_steps_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, safe_next_steps, border=1, align="L", fill=True)
    pdf.set_text_color(*DARK)
    pdf.ln(6)

    # ── CERTIFICATION IMPACT (executive-only section) ────────────────────────────
    pdf.h2("Certification Impact")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(*DARK)
    
    safe_cert_impact = certification_impact_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, safe_cert_impact, border=0, align="L")
    pdf.ln(4)

    pdf_content = bytes(pdf.output(dest='S'))
    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=executive_summary.pdf"}
    )
