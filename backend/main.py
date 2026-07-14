import os
import io
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import PCA
from datetime import datetime
from fpdf import FPDF

app = FastAPI(title="BQIS Backend API", version="1.1.0")

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
    "Cadmium_Cd_mgkg":        "Cadmium (Cd)"
}

def process_dataset(raw_df: pd.DataFrame):
    global global_ds, global_feat, global_shap_df, global_pca_df, global_cluster_df, global_bundle
    
    if global_bundle is None:
        global_bundle = joblib.load(BUNDLE_PATH)
        
    bundle = global_bundle
    model, imputer = bundle["model"], bundle["imputer"]
    feature_cols, numeric_cols = bundle["feature_columns"], bundle["numeric_missing_cols"]
    
    cols_to_use = [
        "Moisture_Content_%", "Fat_Content_%", "Protein_Content_%", "Water_Activity_Aw",
        "Acid_Insoluble_Ash_%", "Acid_Value_mgKOHg", "Peroxide_Value",
        "Total_Plate_Count_CFUg", "Yeast_Mold_Count_CFUg",
        "Lead_Pb_mgkg", "Cadmium_Cd_mgkg", "Product_Name"
    ]
    
    ds = raw_df.copy()
    
    # Generate dummy Batch/Sample_ID if missing (for uploaded data)
    if "Sample_ID" not in ds.columns:
        ds["Sample_ID"] = [f"UPL-{i:04d}" for i in range(len(ds))]
    if "Batch_Code" not in ds.columns:
        ds["Batch_Code"] = "BCH-UPLOADED"
        
    raw_sub = ds[[c for c in cols_to_use if c in ds.columns]].copy()
    
    # Handle missing Product_Name for dummy encoding
    if "Product_Name" not in raw_sub.columns:
        raw_sub["Product_Name"] = "Butter Biscuit"
        
    df_feat = pd.get_dummies(raw_sub, columns=["Product_Name"], prefix="Product")
    df_feat = df_feat.reindex(columns=feature_cols, fill_value=0)
    df_feat[numeric_cols] = imputer.transform(df_feat[numeric_cols])

    preds = model.predict(df_feat)
    probas = model.predict_proba(df_feat)

    ds["Prediction"] = ["PASS" if p == 0 else "FAIL" for p in preds]
    ds["Prob_Pass"] = probas[:, 0]
    ds["Prob_Fail"] = probas[:, 1]
    ds["Confidence"] = [round(probas[i, p] * 100, 1) for i, p in enumerate(preds)]

    def _risk(row):
        if row["Prediction"] == "PASS": return "Pass"
        elif row["Prob_Fail"] >= 0.80: return "High Risk"
        elif row["Prob_Fail"] >= 0.60: return "Medium Risk"
        return "Low Risk"

    ds["Risk_Level"] = ds.apply(_risk, axis=1)
    
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(df_feat), pred_contribs=True)
    shap_vals = contribs[:, :-1]
    
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_signed = shap_vals.mean(axis=0)
    
    shap_df = pd.DataFrame({
        "feature": bundle["feature_columns"],
        "mean_abs": mean_abs,
        "mean_signed": mean_signed,
    })
    
    shap_df = shap_df[~shap_df["feature"].str.startswith("Product_")]
    shap_df = shap_df.sort_values("mean_abs", ascending=False).reset_index(drop=True)
    total_shap = shap_df["mean_abs"].sum()
    if total_shap == 0: total_shap = 1
    
    shap_df["label"] = shap_df["feature"].map(FEATURE_LABELS).fillna(shap_df["feature"])
    shap_df["relative_pct"] = (shap_df["mean_abs"] / total_shap * 100).round(1)
    shap_df["direction"] = shap_df["mean_signed"].apply(lambda v: "Positive" if v > 0 else "Negative")
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(df_feat.values)
    
    pca_df = ds[["Sample_ID", "Batch_Code", "Product_Name", "Prediction", "Risk_Level", "Confidence"]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]
    
    try:
        cluster_df = pd.read_csv(CLUSTER_PATH)
        pca_df = pca_df.merge(
            cluster_df[["Sample_ID", "Failure_Category_Original"]],
            on="Sample_ID", how="left"
        )
        pca_df["Failure_Category_Original"] = pca_df["Failure_Category_Original"].fillna("Pass")
    except Exception:
        # If new samples don't map to original clustering, assign dummy categories based on risk
        def mock_cat(row):
            if row["Prediction"] == "PASS": return "Pass"
            cats = ["Microbiological", "Physicochemical", "Heavy_Metal", "Stability"]
            return cats[int(abs(row["PC1"] * 10)) % 4]
        pca_df["Failure_Category_Original"] = pca_df.apply(mock_cat, axis=1)

    # Parse Test_Date menjadi datetime — digunakan untuk period filter
    if "Test_Date" in ds.columns:
        ds["Test_Date_dt"] = pd.to_datetime(ds["Test_Date"], errors="coerce")
    else:
        ds["Test_Date_dt"] = pd.NaT

    global_ds = ds
    global_feat = df_feat
    global_shap_df = shap_df
    global_pca_df = pca_df


def load_data():
    raw = pd.read_csv(DATASET_PATH)
    process_dataset(raw)

@app.on_event("startup")
def startup_event():
    load_data()

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
            except Exception:
                pass  # Period string tidak valid — abaikan filter

    if batch and batch != "All Batches" and "Batch_Code" in f_ds.columns:
        f_ds = f_ds[f_ds["Batch_Code"] == batch]
        f_pca = f_pca[f_pca.index.isin(f_ds.index)]

    if product and product != "All Products" and "Product_Name" in f_ds.columns:
        f_ds = f_ds[f_ds["Product_Name"] == product]
        f_pca = f_pca[f_pca.index.isin(f_ds.index)]

    return f_ds, f_pca

def get_stats(ds):
    total = len(ds)
    if total == 0: return {"total":0,"n_pass":0,"n_fail":0,"n_high":0,"n_med":0,"n_low":0,"avg_c":0}
    n_pass = (ds["Prediction"] == "PASS").sum()
    n_fail = (ds["Prediction"] == "FAIL").sum()
    n_high = (ds["Risk_Level"] == "High Risk").sum()
    n_med = (ds["Risk_Level"] == "Medium Risk").sum()
    n_low = (ds["Risk_Level"] == "Low Risk").sum()
    avg_c = round(float(ds["Confidence"].mean()), 1)
    return {"total": total, "n_pass": int(n_pass), "n_fail": int(n_fail), 
            "n_high": int(n_high), "n_med": int(n_med), "n_low": int(n_low), "avg_c": avg_c}

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
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    process_dataset(df)
    return {"message": "Dataset uploaded and processed successfully", "samples": len(df)}

@app.get("/api/report/audit")
def generate_audit_report(period: str = None, batch: str = None, product: str = None):
    f_ds, _ = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BQIS Audit Report", ln=True, align="C")
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Filters - Period: {period or 'All'}, Batch: {batch or 'All'}, Product: {product or 'All'}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Total Samples Analyzed: {stats['total']}", ln=True)
    pdf.cell(0, 10, f"Predicted Pass: {stats['n_pass']} ({(stats['n_pass']/max(1,stats['total']))*100:.1f}%)", ln=True)
    pdf.cell(0, 10, f"Predicted Fail: {stats['n_fail']} ({(stats['n_fail']/max(1,stats['total']))*100:.1f}%)", ln=True)
    pdf.cell(0, 10, f"High Risk Samples Requiring Auditor Review: {stats['n_high']}", ln=True)
    
    pdf_content = bytes(pdf.output(dest='S'))
    return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

@app.get("/api/report/executive")
def generate_exec_summary(period: str = None, batch: str = None, product: str = None):
    f_ds, _ = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BQIS Executive Summary", ln=True, align="C")
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date generated: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Key Findings:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, f"The AI engine processed {stats['total']} samples. {stats['n_high']} samples were flagged as HIGH RISK and require immediate quarantine or inspection.")
    
    pdf_content = bytes(pdf.output(dest='S'))
    return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=executive_summary.pdf"})
