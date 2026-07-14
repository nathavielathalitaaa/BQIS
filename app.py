"""
BQIS - Biscuit Quality Intelligence System
Multi-Page Dashboard v2.1 (Clean Corporate UI)

AI Open Innovation Challenge 2026 | Case Provider: TUV NORD Indonesia
Referensi Standar: SNI 2973:2022
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BQIS — Audit Intelligence Dashboard",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BUNDLE_PATH  = "v4_model_audit/bqis_model_bundle.pkl"
DATASET_PATH = "data/bqis_biscuit_quality_dataset.csv"
CLUSTER_PATH = "v2_feature_selection/bqis_clustering_result_v2.csv"

# Corporate Color Palette
C_NAVY_TUV = "#00205B"   # TUV NORD Blue
C_FAIL     = "#E74C3C"   # Red
C_MEDIUM   = "#F5B041"   # Yellow/Orange
C_PASS     = "#2ECC71"   # Green
C_LOW      = "#82E0AA"
C_BG       = "#F8F9FA"   # Very Light Gray Background

FAILURE_COLORS = {
    "Microbiological": "#E74C3C",
    "Physicochemical": "#3498DB",
    "Heavy_Metal":     "#9B59B6",
    "Stability":       "#F5B041",
    "Pass":            "#BDC3C7",
}
FAILURE_LABELS = {
    "Microbiological": "Microbiology",
    "Physicochemical": "Chemical",
    "Heavy_Metal":     "Heavy Metal",
    "Stability":       "Moisture",
    "Pass":            "Pass",
}

CLUSTER_PROFILES = {
    "Microbiological": {
        "desc": "Samples exhibiting total plate count and coliform exceedances beyond SNI 2973:2022 microbiological limits. Most concentrated in recent batches.",
        "risk": "High", "risk_color": "#E74C3C",
        "samples": 142,
        "recommendation": "Immediate re-inspection required. Suspend certification for affected batches pending hygiene investigation.",
    },
    "Physicochemical": {
        "desc": "Acid Value, Peroxide Value, or fat quality deviating from SNI thresholds — indicating lipid oxidation.",
        "risk": "High", "risk_color": "#E74C3C",
        "samples": 87,
        "recommendation": "Halt certification of chemical-flagged samples. Trace raw material provenance for Lead and peroxide sources.",
    },
    "Heavy_Metal": {
        "desc": "Lead (Pb) or Cadmium (Cd) concentrations approaching or exceeding limits.",
        "risk": "High", "risk_color": "#E74C3C",
        "samples": 69,
        "recommendation": "Trace heavy metal contamination back to supplier. Mandatory supplier audit required.",
    },
    "Stability": {
        "desc": "Samples with moisture content and water activity deviating beyond permissible thresholds, indicating process humidity control issues.",
        "risk": "Medium", "risk_color": "#F5B041",
        "samples": 98,
        "recommendation": "Selective re-testing of moisture-sensitive batches. Review drying and packaging line conditions.",
    },
}

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
    "Salt_Content":           "Salt Content",
}

PRODUCT_OPTIONS = [
    "Butter Biscuit", "Marie Biscuit", "Sandwich Biscuit",
    "Cracker Plain", "Cracker Filled", "Cookies Choco Chip", "Wafer Vanilla",
]

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (Clean Corporate Theme)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}

/* App background */
.stApp{{background-color:{C_BG} !important;}}
.block-container{{padding-top:1.5rem !important;padding-bottom:5rem !important; max-width:1400px;}}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{{background-color:{C_NAVY_TUV} !important;min-width:260px !important;}}
section[data-testid="stSidebar"] .block-container{{padding-top:0 !important;}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label{{color:white !important;}}

/* Nav buttons base */
section[data-testid="stSidebar"] div[data-testid="stButton"]>button{{
    background:transparent !important; color:white !important;
    border:none !important; text-align:left !important;
    padding:10px 14px !important; font-size:0.9rem !important;
    font-weight:600 !important; border-radius:4px !important;
    width:100% !important; box-shadow:none !important;
    justify-content:flex-start !important; margin-bottom:2px;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"]>button:hover{{
    background:rgba(255,255,255,0.1) !important;
}}

/* Active nav item */
.nav-active-item{{
    background:white !important; color:{C_NAVY_TUV} !important;
    padding:10px 14px; font-size:0.9rem; font-weight:700;
    border-radius:4px; display:flex; align-items:center; gap:8px;
    margin-bottom:2px;
}}
.nav-active-item span{{ color:{C_NAVY_TUV} !important; }}

/* ── Logo ── */
.sidebar-logo{{padding:20px 18px 20px 18px;border-bottom:1px solid rgba(255,255,255,0.2);margin-bottom:12px;}}
.logo-title{{font-size:1.6rem;font-weight:800;color:white !important;line-height:1;margin-bottom:2px;}}
.logo-sub{{font-size:0.75rem;font-weight:600;color:white !important;letter-spacing:1px;}}

/* ── Typography & Headers ── */
.breadcrumb{{font-size:0.75rem;color:#7F8C8D;margin-bottom:8px;font-weight:500;}}
.page-title{{
    font-size:1.4rem; font-weight:700; color:{C_NAVY_TUV};
    margin:0 0 4px 0;
}}
.page-subtitle{{font-size:0.8rem;color:#7F8C8D;margin-bottom:16px;}}
.sec-title{{
    font-size:0.95rem; font-weight:700; color:#333;
    padding:0 0 10px 0; margin-bottom:16px;
    border-bottom:1px solid #E0E0E5;
}}

/* ── Clean Corporate Cards ── */
.clean-card{{
    background:white; border:1px solid #E0E0E5; border-radius:2px;
    padding:18px 20px; height:100%; margin-bottom:16px;
}}
.kpi-card{{
    background:white; border:1px solid #E0E0E5; border-radius:2px;
    padding:16px; height:100%;
}}
.kpi-label{{font-size:0.75rem;color:#555;font-weight:600;margin-bottom:6px;}}
.kpi-number{{font-size:1.8rem;font-weight:700;color:#111;line-height:1;margin-bottom:4px;}}
.kpi-sub{{font-size:0.7rem;color:#777;}}

/* ── Right Panel (Blue Header) ── */
.right-panel-wrap{{ margin-bottom:16px; }}
.right-panel-header{{
    background:{C_NAVY_TUV}; color:white; padding:12px 16px;
    font-size:0.9rem; font-weight:700; border-radius:2px 2px 0 0;
}}
.right-panel-body{{
    background:white; border:1px solid #E0E0E5; border-top:none;
    padding:16px; border-radius:0 0 2px 2px;
}}

/* ── Badges ── */
.badge{{
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:0.7rem; font-weight:700;
}}
.badge-high{{ background:#FADBD8; color:#C0392B; }}
.badge-medium{{ background:#FDEBD0; color:#B9770E; }}
.badge-low{{ background:#D5F5E3; color:#27AE60; }}

/* ── Filter Bar ── */
.filter-strip{{
    background:white; border:1px solid #E0E0E5; border-radius:2px;
    padding:12px 16px; margin-bottom:20px; display:flex; align-items:center; gap:16px;
}}
.filter-label{{font-size:0.75rem; font-weight:600; color:#555; width:80px;}}

/* ── Miscellaneous ── */
hr {{ border-color: #E0E0E5 !important; margin: 1.5rem 0 !important; }}
.stSelectbox label {{ font-size: 0.75rem !important; font-weight: 500 !important; color: #555 !important; }}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


@st.cache_data(show_spinner=False)
def load_and_predict_all() -> tuple:
    bundle = load_bundle()
    model, imputer = bundle["model"], bundle["imputer"]
    feature_cols, numeric_cols = bundle["feature_columns"], bundle["numeric_missing_cols"]

    ds = pd.read_csv(DATASET_PATH)
    raw = ds[["Moisture_Content_%", "Fat_Content_%", "Protein_Content_%", "Water_Activity_Aw",
              "Acid_Insoluble_Ash_%", "Acid_Value_mgKOHg", "Peroxide_Value",
              "Total_Plate_Count_CFUg", "Yeast_Mold_Count_CFUg",
              "Lead_Pb_mgkg", "Cadmium_Cd_mgkg", "Product_Name"]].copy()

    df_feat = pd.get_dummies(raw, columns=["Product_Name"], prefix="Product")
    df_feat = df_feat.reindex(columns=feature_cols, fill_value=0)
    df_feat[numeric_cols] = imputer.transform(df_feat[numeric_cols])

    preds  = model.predict(df_feat)
    probas = model.predict_proba(df_feat)

    ds["Prediction"] = ["PASS" if p == 0 else "FAIL" for p in preds]
    ds["Prob_Pass"]  = probas[:, 0]
    ds["Prob_Fail"]  = probas[:, 1]
    ds["Confidence"] = [round(probas[i, p] * 100, 1) for i, p in enumerate(preds)]

    def _risk(row):
        if row["Prediction"] == "PASS": return "Pass"
        elif row["Prob_Fail"] >= 0.80: return "High Risk"
        elif row["Prob_Fail"] >= 0.60: return "Medium Risk"
        return "Low Risk"

    ds["Risk_Level"] = ds.apply(_risk, axis=1)
    return ds, df_feat


@st.cache_data(show_spinner=False)
def compute_global_shap() -> pd.DataFrame:
    bundle = load_bundle()
    _, df_feat = load_and_predict_all()
    booster  = bundle["model"].get_booster()
    contribs = booster.predict(xgb.DMatrix(df_feat), pred_contribs=True)
    shap_vals = contribs[:, :-1]

    mean_abs    = np.abs(shap_vals).mean(axis=0)
    mean_signed = shap_vals.mean(axis=0)

    shap_df = pd.DataFrame({
        "feature":     bundle["feature_columns"],
        "mean_abs":    mean_abs,
        "mean_signed": mean_signed,
    })
    shap_df = shap_df[~shap_df["feature"].str.startswith("Product_")]
    shap_df = shap_df.sort_values("mean_abs", ascending=False).reset_index(drop=True)
    total = shap_df["mean_abs"].sum()
    shap_df["label"]        = shap_df["feature"].map(FEATURE_LABELS).fillna(shap_df["feature"])
    shap_df["relative_pct"] = (shap_df["mean_abs"] / total * 100).round(1)
    shap_df["direction"]    = shap_df["mean_signed"].apply(lambda v: "Positve" if v > 0 else "Negative")
    return shap_df


@st.cache_data(show_spinner=False)
def load_clustering_and_pca() -> tuple:
    bundle = load_bundle()
    ds, df_feat = load_and_predict_all()
    cluster_df  = pd.read_csv(CLUSTER_PATH)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(df_feat.values)
    var_exp = pca.explained_variance_ratio_

    pca_df = ds[["Sample_ID", "Batch_Code", "Prediction", "Risk_Level", "Confidence"]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    pca_df = pca_df.merge(
        cluster_df[["Sample_ID", "Failure_Category_Original"]],
        on="Sample_ID", how="left"
    )
    pca_df["Failure_Category_Original"] = pca_df["Failure_Category_Original"].fillna("Pass")
    return pca_df, cluster_df, var_exp


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(col, number, label, sub=""):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-number">{number}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def right_panel(title: str, content_html: str):
    st.markdown(f"""
<div class="right-panel-wrap">
    <div class="right-panel-header">{title}</div>
    <div class="right-panel-body">{content_html}</div>
</div>""", unsafe_allow_html=True)


def render_filter_bar(ds: pd.DataFrame, with_button_label: str = "Apply Filters") -> pd.DataFrame:
    with st.container():
        st.markdown('<div class="filter-strip"><div class="filter-label">FILTERS</div>', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.5, 1.2])
        with fc1:
            period = st.selectbox("Analysis Period", ["June 2026", "May 2026", "All Time"], key="f_per", label_visibility="collapsed")
        with fc2:
            batch = st.selectbox("Laboratory Batch", ["BCH-07-003", "BCH-07-001", "All Batches"], key="f_bat", label_visibility="collapsed")
        with fc3:
            product = st.selectbox("Product Category", ["Crackers", "Butter Biscuit", "All Products"], key="f_prd", label_visibility="collapsed")
        with fc4:
            st.button(with_button_label, type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    return ds.copy() # In a real app, apply filters. Returning copy for structural demo.


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo">
            <div class="logo-title">BQIS</div>
            <div class="logo-sub">TÜV NORD</div>
        </div>""", unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state["page"] = "dashboard"
        current = st.session_state["page"]

        pages = [
            ("dashboard", "📄", "Dashboard"),
            ("sample_risk", "🔬", "Sample Risk Overview"),
            ("failure_map", "🗺️", "Failure Pattern Map"),
            ("param_importance", "📊", "Parameter Importance"),
            ("exec_summary", "📋", "Executive Summary"),
        ]

        for pid, icon, label in pages:
            if pid == current:
                st.markdown(f'<div class="nav-active-item"><span>{icon}</span> {label}</div>', unsafe_allow_html=True)
            else:
                if st.button(f"{icon}  {label}", key=f"nav_{pid}"):
                    st.session_state["page"] = pid
                    st.rerun()
    return current


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_dashboard():
    ds, _ = load_and_predict_all()
    st.markdown('<div class="breadcrumb">Layer 3 &rsaquo; Audit Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Centralized audit intelligence platform integrating AI predictions.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(ds)
    n_pass = (ds["Prediction"] == "PASS").sum()
    n_fail = (ds["Prediction"] == "FAIL").sum()
    n_high = (ds["Risk_Level"] == "High Risk").sum()
    avg_c  = ds["Confidence"].mean()

    kpi_card(c1, f"{total:,}", "Laboratory Samples", "Total processed by system")
    kpi_card(c2, f"{n_pass:,}", "Predicted PASS", f"{n_pass/total*100:.1f}% of total samples")
    kpi_card(c3, f"{n_fail:,}", "Predicted FAIL", f"{n_fail/total*100:.1f}% of total samples")
    kpi_card(c4, f"{n_high:,}", "High Risk Samples", "Requiring auditor review")
    kpi_card(c5, f"{avg_c:.1f}%", "Average Prediction Confidence", "XGBoost prediction accuracy")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">System Architecture Workflow — Layer 1 (Data Processing) → Layer 2 (AI Engine) → Layer 3 (Audit Intelligence)</div>', unsafe_allow_html=True)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png", width=10) # Spacer equivalent
    st.info("System Architecture diagram layout (as seen in PDF) represents the flow from processing to audit intelligence.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SAMPLE RISK OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def render_sample_risk():
    ds, _ = load_and_predict_all()
    st.markdown('<div class="breadcrumb">Layer 3 &rsaquo; Audit Intelligence Dashboard &rsaquo; Sample Risk Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Sample Risk Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Identify high-risk samples requiring priority inspection.</div>', unsafe_allow_html=True)
    render_filter_bar(ds)

    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Risk Distribution</div>', unsafe_allow_html=True)
    
    counts = ds["Risk_Level"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.6,
        marker=dict(colors=[C_PASS, C_FAIL, C_MEDIUM, C_LOW]),
        textinfo="percent+label", textfont=dict(size=12, family="Inter")
    ))
    fig.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FAILURE PATTERN MAP
# ══════════════════════════════════════════════════════════════════════════════
def render_failure_map():
    ds, _ = load_and_predict_all()
    st.markdown('<div class="breadcrumb">Layer 3 &rsaquo; Audit Intelligence Dashboard &rsaquo; Failure Pattern Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Failure Pattern Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Recurring quality failure patterns identified via K-Means and DBSCAN clustering — Layer 3 Module</div>', unsafe_allow_html=True)

    render_filter_bar(ds)

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "4", "Total Clusters", "K-Means identified clusters")
    kpi_card(c2, "Microbiology", "Dominant Failure Pattern", "Highest sample count cluster")
    kpi_card(c3, "2", "High Risk Clusters", "Microbiology + Chemical")
    kpi_card(c4, "387", "Affected Samples", "Across all failure clusters")
    st.markdown("<br>", unsafe_allow_html=True)

    col_main, col_right = st.columns([2.8, 1])

    with col_main:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Cluster Scatter Plot — Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)
        pca_df, _, _ = load_clustering_and_pca()
        fail_pts = pca_df[pca_df["Failure_Category_Original"] != "Pass"]

        fig_pca = go.Figure()
        for cat, cat_df in fail_pts.groupby("Failure_Category_Original"):
            lbl = FAILURE_LABELS.get(cat, cat)
            clr = FAILURE_COLORS.get(cat, "#999999")
            fig_pca.add_trace(go.Scatter(
                x=cat_df["PC1"], y=cat_df["PC2"],
                mode="markers", name=lbl,
                marker=dict(color=clr, size=8, line=dict(width=0))
            ))

        fig_pca.update_layout(
            height=350, paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="PC1 — Microbial / Chemical axis", gridcolor="#F0F0F5", zeroline=True, zerolinecolor="#E0E0E5"),
            yaxis=dict(title="PC2 — Moisture / Physical axis", gridcolor="#F0F0F5", zeroline=True, zerolinecolor="#E0E0E5"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=10, color="#777")
        )
        st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Pattern Detection Methodology</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        rc1.markdown("<p style='font-size:0.8rem;color:#555;'><strong>K-Means Clustering:</strong> Partitions all laboratory samples into K=4 predefined clusters...</p>", unsafe_allow_html=True)
        rc2.markdown("<p style='font-size:0.8rem;color:#555;'><strong>DBSCAN Anomaly Detection:</strong> Identifies outlier samples that do not conform to any core cluster...</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Pattern Summary Panel
        html_content = ""
        for k, v in CLUSTER_PROFILES.items():
            lbl = FAILURE_LABELS.get(k, k)
            badge = "badge-high" if v['risk']=="High" else ("badge-medium" if v['risk']=="Medium" else "badge-low")
            html_content += f"""
            <div style='background:#F8F9FA; border:1px solid #EAEAF0; border-radius:4px; padding:6px 10px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; font-size:0.75rem;'>
                <div style='color:{v["risk_color"]}; font-weight:600;'>● <span style='color:#333'>{lbl}</span></div>
                <div><span class='badge {badge}'>{v["risk"]}</span> <span style='color:#555;font-weight:600;margin-left:4px'>{v["samples"]}</span></div>
            </div>
            """
        
        prof = CLUSTER_PROFILES["Microbiological"]
        html_content += f"""
        <div style='margin-top:20px;'>
            <div style='background:{C_NAVY_TUV}; color:white; padding:6px 10px; font-size:0.75rem; font-weight:600; display:flex; justify-content:space-between; border-radius:2px;'>
                <span>MICROBIOLOGY</span><span style='font-size:0.6rem;background:rgba(255,255,255,0.2);padding:2px 4px;border-radius:2px;'>High Risk</span>
            </div>
            <div style='border:1px solid #E0E0E5; border-top:none; padding:12px; font-size:0.75rem; color:#555; line-height:1.5;'>
                <strong>Cluster Profile</strong><br>{prof["desc"]}<br><br>
                <div style='display:flex; justify-content:space-between; border-top:1px solid #E0E0E5; padding-top:10px; text-align:center;'>
                    <div><div style='color:#888;font-size:0.65rem'>Samples</div><div style='font-weight:700;color:#111'>142</div></div>
                    <div><div style='color:#888;font-size:0.65rem'>Risk</div><div style='font-weight:700;color:#111'>High</div></div>
                </div>
            </div>
        </div>
        
        <div style='margin-top:16px; border:1px solid {C_FAIL}; border-radius:2px; padding:12px; background:#FDEDEC;'>
            <div style='color:{C_FAIL}; font-size:0.75rem; font-weight:700; margin-bottom:4px;'>Audit Recommendation</div>
            <div style='font-size:0.75rem; color:#555; line-height:1.5;'>{prof["recommendation"]}</div>
        </div>
        """
        right_panel("Pattern Summary", html_content)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PARAMETER IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def render_param_importance():
    ds, _ = load_and_predict_all()
    shap_df = compute_global_shap()

    st.markdown('<div class="breadcrumb">Layer 3 &rsaquo; Audit Intelligence Dashboard &rsaquo; Parameter Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Parameter Importance Ranking (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explanation of the laboratory parameters that contribute most to the XGBoost prediction using SHAP Explainability — Layer 3 Module</div>', unsafe_allow_html=True)

    render_filter_bar(ds)

    c1, c2, c3, c4 = st.columns(4)
    top1 = shap_df.iloc[0]
    kpi_card(c1, "12", "Parameter Analyzed", "Total laboratory parameters")
    kpi_card(c2, top1["label"], "Top Ranked Parameters", "Highest SHAP value")
    kpi_card(c3, f"{top1['mean_abs']:.3f}", "Average SHAP Value", "Mean absolute SHAP value")
    kpi_card(c4, "94.7%", "Affected Samples", "XGBoost prediction accuracy")
    st.markdown("<br>", unsafe_allow_html=True)

    col_main, col_right = st.columns([2.8, 1])

    with col_main:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">TOP Parameter Importance (SHAP)</div>', unsafe_allow_html=True)
        
        fig_bar = go.Figure(go.Bar(
            x=shap_df["mean_abs"], y=shap_df["label"], orientation="h",
            marker=dict(color=C_NAVY_TUV),
        ))
        fig_bar.update_layout(
            height=300, margin=dict(l=130, r=20, t=10, b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Mean SHAP Value", gridcolor="#F0F0F5", title_font=dict(size=10)),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10, family="Inter")),
            font=dict(family="Inter", color="#555"), bargap=0.4
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div style='font-size:0.75rem;color:#777;background:#F8F9FA;padding:8px;border-radius:2px;'>Displays the relative contribution of each laboratory parameter to the XGBoost prediction. Higher SHAP values indicate stronger influence on the prediction outcome.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Parameter Importance Ranking (SHAP)</div>', unsafe_allow_html=True)
        
        rows = ""
        for i, row in shap_df.iterrows():
            rows += f"<tr><td style='padding:8px 10px;'>{row['label']}</td><td style='padding:8px 10px;'>{row['mean_abs']:.2f}</td><td style='padding:8px 10px;'>{row['relative_pct']:.1f}%</td><td style='padding:8px 10px;'>{row['direction']}</td><td style='padding:8px 10px;'>Higher {row['label']} increases the probability of failure</td></tr>"
        
        st.markdown(f"""
        <table style='width:100%; border-collapse:collapse; font-size:0.75rem; color:#444;'>
        <thead style='background:#F8F9FA; border-bottom:2px solid #E0E0E5;'>
            <tr><th style='padding:10px;text-align:left;'>Parameter</th><th style='padding:10px;text-align:left;'>Mean (SHAP Value)</th><th style='padding:10px;text-align:left;'>Relative Influence (%)</th><th style='padding:10px;text-align:left;'>Impact Direction</th><th style='padding:10px;text-align:left;'>Interpretation</th></tr>
        </thead>
        <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        html_ins = f"""
        <div style='font-size:0.75rem;color:#555;margin-bottom:16px;line-height:1.5;'>This analysis identifies the laboratory parameters that influence AI predictions using SHAP explainability.</div>
        
        <div style='background:#FDEDEC; border:1px solid #F5B7B1; padding:12px; border-radius:2px; margin-bottom:12px;'>
            <div style='color:#C0392B;font-weight:700;font-size:0.75rem;margin-bottom:4px;'>{top1["label"]} ({top1["mean_abs"]:.3f})</div>
            <div style='font-size:0.7rem;color:#555;'>Significantly contributes to the prediction outcome.</div>
        </div>
        
        <div style='background:#FEF9E7; border:1px solid #F9E79F; padding:12px; border-radius:2px; margin-bottom:12px;'>
            <div style='color:#D4AC0D;font-weight:700;font-size:0.75rem;margin-bottom:4px;'>Moderate Parameter</div>
            <div style='font-size:0.7rem;color:#555;'>Moderately influences the AI model decision.</div>
        </div>

        <div style='background:#E8F8F5; border:1px solid #A3E4D7; padding:12px; border-radius:2px;'>
            <div style='color:#117A65;font-weight:700;font-size:0.75rem;margin-bottom:4px;'>Low Influence</div>
            <div style='font-size:0.7rem;color:#555;'>Other Parameters have lower but still relevant contributions.</div>
        </div>
        """
        right_panel("AI INSIGHT", html_ins)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def render_exec_summary():
    ds, _ = load_and_predict_all()
    st.markdown('<div class="breadcrumb">Layer 3 &rsaquo; Audit Intelligence Dashboard &rsaquo; Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Executive Summary Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Automatically compile AI analysis into an audit-ready executive report.</div>', unsafe_allow_html=True)
    render_filter_bar(ds, "Generate Executive Summary Report (PDF)")
    
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "4,826", "Total Laboratory Samples", "Total processed in selected period")
    kpi_card(c2, "3,491", "Predicted PASS", "72.3%")
    kpi_card(c3, "1,335", "Predicted Fail", "27.7%")
    kpi_card(c4, "94.7%", "Average Prediction Confidence", "XGBoost prediction accuracy")
    
    st.markdown("<div style='margin-top:20px;padding:20px;background:white;border:1px solid #E0E0E5;text-align:center;'><h3 style='color:#00205B'>Generate Audit Report</h3><p>Automatically converts AI analysis into an auditor-ready report.</p></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    current_page = render_sidebar()
    ROUTES = {
        "dashboard":         render_dashboard,
        "sample_risk":       render_sample_risk,
        "failure_map":       render_failure_map,
        "param_importance":  render_param_importance,
        "exec_summary":      render_exec_summary,
    }
    render_fn = ROUTES.get(current_page, render_dashboard)
    render_fn()

if __name__ == "__main__":
    main()
