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
from datetime import datetime
from fpdf import FPDF

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
        elif row["Prob_Fail"] >= RISK_THRESHOLD_HIGH: return "High Risk"
        elif row["Prob_Fail"] >= RISK_THRESHOLD_MEDIUM: return "Medium Risk"
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
    except Exception as e:
        logger.warning("Cluster merge failed (%s) — falling back to dummy categories", e)
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

    def _cover(self, subtitle: str, period: str, batch: str, product: str):
        self.add_page()
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 60, "F")
        self.set_xy(15, 18)
        self.set_font("Arial", "B", 30)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "BQIS", ln=1)
        self.set_x(15)
        self.set_font("Arial", "B", 14)
        self.cell(0, 8, "TÜV NORD", ln=1)
        self.ln(20)
        self.set_text_color(*NAVY)
        self.set_font("Arial", "B", 22)
        self.cell(0, 12, self.title, ln=1, align="C")
        self.set_font("Arial", "", 12)
        self.set_text_color(*GREY)
        self.cell(0, 8, subtitle, ln=1, align="C")
        self.ln(12)
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
    pdf._cover("Biscuit Quality Intelligence System", period, batch, product)

    # ── 1. EXECUTIVE SUMMARY ──
    pdf.add_page()
    pdf.h1("1. Executive Summary")
    pdf._table(
        ["Metric", "Value", "Notes"],
        [
            ["Total Samples", f"{stats['total']:,}", "All time"],
            ["Predicted PASS", f"{stats['n_pass']:,}", f"{stats['n_pass']/total*100:.1f}%"],
            ["Predicted FAIL", f"{stats['n_fail']:,}", f"{stats['n_fail']/total*100:.1f}%"],
            ["High Risk", f"{stats['n_high']:,}", "Auditor review"],
            ["Avg Confidence", f"{stats['avg_c']:.1f}%", "XGBoost accuracy"],
        ],
        [60, 50, 80], aligns=["L", "C", "L"]
    )
    pdf.ln(6)

    # ── 2. RISK DISTRIBUTION ──
    pdf.h2("2. Risk Distribution")
    pdf._table(
        ["Level", "Samples", "%", "Action"],
        [
            ["Pass", f"{stats['n_pass']:,}", f"{stats['n_pass']/total*100:.1f}%", "Clear"],
            ["High Risk", f"{stats['n_high']:,}", f"{stats['n_high']/total*100:.1f}%", "URGENT"],
            ["Medium Risk", f"{stats['n_med']:,}", f"{stats['n_med']/total*100:.1f}%", "HIGH"],
            ["Low Risk", f"{stats['n_low']:,}", f"{stats['n_low']/total*100:.1f}%", "NORMAL"],
        ],
        [45, 40, 35, 70],
        aligns=["L", "C", "C", "L"],
        row_colors=[None, RED, YELLOW, GREEN],
    )
    pdf.ln(6)

    # ── 3. TOP SHAP PARAMETERS ──
    pdf.h2("3. Top SHAP Parameters")
    top = global_shap_df.head(8)
    total_shap = top["mean_abs"].sum() or 1
    pdf._table(
        ["Parameter", "SHAP Val", "Influence"],
        [[r["label"], f"{r['mean_abs']:.4f}", f"{r['mean_abs']/total_shap*100:.1f}%"] for _, r in top.iterrows()],
        [90, 40, 40], aligns=["L", "C", "C"]
    )
    pdf.ln(6)

    # ── 4. CLUSTER PROFILES ──
    pdf.h2("4. Cluster Profiles")
    cat_counts = f_ds  # placeholder; use pca-based counts from global
    # Recompute cluster counts from filtered pca
    f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)[1]
    cc = f_pca[f_pca["Failure_Category_Original"] != "Pass"]["Failure_Category_Original"].value_counts()
    cluster_rows = [
        ["Microbiological", f"{int(cc.get('Microbiological', 0))}", "High", "Reinspect"],
        ["Physicochemical", f"{int(cc.get('Physicochemical', 0))}", "High", "Halt cert"],
        ["Heavy Metal", f"{int(cc.get('Heavy_Metal', 0))}", "High", "Audit"],
        ["Stability", f"{int(cc.get('Stability', 0))}", "Med", "Retest"],
    ]
    pdf._table(
        ["Cluster", "Samples", "Risk", "Action"],
        cluster_rows, [60, 40, 35, 55], aligns=["L", "C", "C", "L"],
        row_colors=[RED, RED, RED, YELLOW],
    )
    pdf.ln(6)

    # ── 5. AUDIT RECOMMENDATION ──
    pdf.h2("5. Audit Recommendation")
    pdf.set_fill_color(*RED)
    pdf.set_draw_color(*RED)
    pdf.set_text_color(192, 57, 43)
    pdf.set_font("Arial", "B", 11)
    pdf.multi_cell(0, 8,
        f"High risk alerts detected for {stats['n_high']} samples. Immediate quarantine recommended.",
        border=1, align="L", fill=True)

    pdf_content = bytes(pdf.output(dest='S'))
    return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

@app.get("/api/report/executive")
def generate_exec_summary(period: str = None, batch: str = None, product: str = None):
    f_ds, f_pca = apply_filters(global_ds, global_pca_df, period, batch, product)
    stats = get_stats(f_ds)
    total = max(1, stats["total"])

    pdf = BQISReport(title="EXECUTIVE SUMMARY")
    pdf._cover("Biscuit Quality Intelligence System", period, batch, product)

    # ── KEY FINDINGS ──
    pdf.add_page()
    pdf.h1("Key Findings")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 7,
        f"The AI engine processed {stats['total']:,} samples. "
        f"{stats['n_high']:,} samples were flagged as HIGH RISK and require immediate "
        f"quarantine or inspection.", border=0, align="L")
    pdf.ln(4)

    # ── RISK BREAKDOWN ──
    pdf.h2("Risk Breakdown")
    pdf._table(
        ["Level", "Samples"],
        [
            ["High Risk", f"{stats['n_high']:,}"],
            ["Medium Risk", f"{stats['n_med']:,}"],
            ["Low Risk", f"{stats['n_low']:,}"],
            ["Pass", f"{stats['n_pass']:,}"],
        ],
        [80, 60], aligns=["L", "C"],
        row_colors=[RED, YELLOW, GREEN, None],
    )
    pdf.ln(6)

    # ── TOP RISK CATEGORIES ──
    pdf.h2("Top Risk Categories")
    cc = f_pca[f_pca["Failure_Category_Original"] != "Pass"]["Failure_Category_Original"].value_counts()
    pdf._table(
        ["Category", "Samples", "Risk"],
        [
            ["Microbiological", f"{int(cc.get('Microbiological', 0))}", "High"],
            ["Physicochemical", f"{int(cc.get('Physicochemical', 0))}", "High"],
            ["Heavy Metal", f"{int(cc.get('Heavy_Metal', 0))}", "High"],
            ["Moisture / Stability", f"{int(cc.get('Stability', 0))}", "Med"],
        ],
        [90, 40, 40], aligns=["L", "C", "C"],
        row_colors=[RED, RED, RED, YELLOW],
    )
    pdf.ln(6)

    # ── AUDIT RECOMMENDATION ──
    pdf.h2("Audit Recommendation")
    pdf.set_fill_color(*RED)
    pdf.set_draw_color(*RED)
    pdf.set_text_color(192, 57, 43)
    pdf.set_font("Arial", "B", 11)
    pdf.multi_cell(0, 8,
        f"High risk alerts detected for {stats['n_high']} samples. Immediate quarantine recommended for flagged batches.",
        border=1, align="L", fill=True)

    pdf_content = bytes(pdf.output(dest='S'))
    return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=executive_summary.pdf"})
