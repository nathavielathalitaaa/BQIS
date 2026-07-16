"""
generate_dataset.py
-------------------
Generates a new BQIS biscuit quality dataset with 3000 rows,
matching the statistical distributions and patterns of the original dataset.

Run from the BQIS root directory:
    python generate_dataset.py
"""

import random
import csv
from datetime import date, timedelta

# ─── Seed for reproducibility ────────────────────────────────────────────────
random.seed(42)

# ─── Constants from original dataset ─────────────────────────────────────────
PRODUCT_NAMES = [
    "Butter Biscuit",
    "Cracker Plain",
    "Cracker Filled",
    "Cookies Choco Chip",
    "Sandwich Biscuit",
    "Wafer Vanilla",
    "Marie Biscuit",
]

PRODUCT_WEIGHTS = [0.13, 0.18, 0.17, 0.14, 0.14, 0.13, 0.11]

FAILURE_CATEGORIES = ["None", "Physicochemical", "Heavy_Metal", "Microbiological", "Stability"]
FAILURE_WEIGHTS    = [0.60,   0.15,              0.10,          0.08,               0.07]

START_DATE = date(2025, 1, 1)
END_DATE   = date(2026, 12, 31)

BATCH_SUFFIXES = list(range(1, 9))

# ─── Helpers ─────────────────────────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def batch_code(d):
    suffix = random.choice(BATCH_SUFFIXES)
    return f"B-{d.strftime('%y%m%d')}-{suffix}"

def sample_param(name, category, dist, overrides):
    mu, sigma, lo, hi, decimals = dist[name]
    if category in overrides and name in overrides[category]:
        mu2, sigma2, lo2, hi2 = overrides[category][name]
        val = random.gauss(mu2, sigma2)
        val = clamp(val, lo2, hi2)
    else:
        val = random.gauss(mu, sigma)
        val = clamp(val, lo, hi)

    if random.random() < 0.05:
        return ""
    return round(val, decimals)

# ─── Per-parameter distributions ─────────────────────────────────────────────
PARAM_DIST = {
    "Moisture_Content_%":     (3.5,  1.2,  0.3,  7.5,  2),
    "Fat_Content_%":          (19.0, 4.0,  8.0,  35.0, 2),
    "Protein_Content_%":      (6.5,  1.5,  2.0,  12.0, 2),
    "Water_Activity_Aw":      (0.55, 0.09, 0.30, 0.85, 3),
    "Acid_Insoluble_Ash_%":   (0.045,0.04, 0.00, 0.20, 3),
    "Acid_Value_mgKOHg":      (1.1,  0.55, 0.00, 3.50, 2),
    "Peroxide_Value":         (1.1,  0.55, 0.00, 3.50, 2),
    "Total_Plate_Count_CFUg": (6000, 12000, 50, 100000, 0),
    "Yeast_Mold_Count_CFUg":  (2000, 4000,  10,  50000, 0),
    "Lead_Pb_mgkg":           (0.18, 0.14, 0.00, 0.80, 3),
    "Cadmium_Cd_mgkg":        (0.07, 0.07, 0.00, 0.35, 3),
}

CATEGORY_OVERRIDES = {
    "Physicochemical": {
        "Moisture_Content_%":   (5.8,  0.8,  4.5,  7.5),
        "Water_Activity_Aw":    (0.72, 0.07, 0.60, 0.87),
        "Acid_Insoluble_Ash_%": (0.13, 0.05, 0.08, 0.22),
        "Protein_Content_%":    (4.0,  1.5,  1.5,  6.0),
    },
    "Microbiological": {
        "Total_Plate_Count_CFUg": (55000, 25000, 20000, 120000),
        "Yeast_Mold_Count_CFUg":  (18000, 10000,  5000,  65000),
        "Water_Activity_Aw":      (0.70,  0.06,   0.60,   0.85),
    },
    "Heavy_Metal": {
        "Lead_Pb_mgkg":    (0.52, 0.15, 0.35, 0.85),
        "Cadmium_Cd_mgkg": (0.22, 0.08, 0.15, 0.38),
    },
    "Stability": {
        "Acid_Value_mgKOHg": (2.40, 0.35, 1.80, 3.80),
        "Peroxide_Value":    (2.30, 0.40, 1.80, 3.80),
        "Moisture_Content_%":(5.5,  0.9,  4.0,  7.5),
    },
}

# ─── Main generation ──────────────────────────────────────────────────────────
N = 3000
OUTPUT_PATH = "data/bqis_biscuit_quality_dataset_3000.csv"

COLUMNS = [
    "Sample_ID", "Batch_Code", "Product_Name", "Test_Date",
    "Moisture_Content_%", "Fat_Content_%", "Protein_Content_%",
    "Water_Activity_Aw", "Acid_Insoluble_Ash_%", "Acid_Value_mgKOHg",
    "Peroxide_Value", "Total_Plate_Count_CFUg", "Yeast_Mold_Count_CFUg",
    "Lead_Pb_mgkg", "Cadmium_Cd_mgkg",
    "Failure_Category", "Historical_Status",
]

rows = []
used_ids = set()

for i in range(N):
    while True:
        sample_num = random.randint(1, 9999)
        sid = f"SPL-2026-{sample_num:04d}"
        if sid not in used_ids:
            used_ids.add(sid)
            break

    test_date   = random_date(START_DATE, END_DATE)
    batch       = batch_code(test_date)
    product     = random.choices(PRODUCT_NAMES, weights=PRODUCT_WEIGHTS)[0]
    category    = random.choices(FAILURE_CATEGORIES, weights=FAILURE_WEIGHTS)[0]
    hist_status = "Pass" if category == "None" else "Fail"
    fail_col    = category

    moisture  = sample_param("Moisture_Content_%",     category, PARAM_DIST, CATEGORY_OVERRIDES)
    fat       = sample_param("Fat_Content_%",          category, PARAM_DIST, CATEGORY_OVERRIDES)
    protein   = sample_param("Protein_Content_%",      category, PARAM_DIST, CATEGORY_OVERRIDES)
    water_act = sample_param("Water_Activity_Aw",      category, PARAM_DIST, CATEGORY_OVERRIDES)
    ash       = sample_param("Acid_Insoluble_Ash_%",   category, PARAM_DIST, CATEGORY_OVERRIDES)
    acid_val  = sample_param("Acid_Value_mgKOHg",      category, PARAM_DIST, CATEGORY_OVERRIDES)
    peroxide  = sample_param("Peroxide_Value",         category, PARAM_DIST, CATEGORY_OVERRIDES)
    tpc       = sample_param("Total_Plate_Count_CFUg", category, PARAM_DIST, CATEGORY_OVERRIDES)
    ymc       = sample_param("Yeast_Mold_Count_CFUg",  category, PARAM_DIST, CATEGORY_OVERRIDES)
    lead      = sample_param("Lead_Pb_mgkg",           category, PARAM_DIST, CATEGORY_OVERRIDES)
    cadmium   = sample_param("Cadmium_Cd_mgkg",        category, PARAM_DIST, CATEGORY_OVERRIDES)

    rows.append([
        sid, batch, product, test_date.strftime("%Y-%m-%d"),
        moisture, fat, protein, water_act, ash, acid_val,
        peroxide, tpc, ymc, lead, cadmium,
        fail_col, hist_status,
    ])

rows.sort(key=lambda r: (r[3], r[0]))

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    writer.writerows(rows)

pass_count = sum(1 for r in rows if r[-1] == "Pass")
fail_count = N - pass_count
cat_counts = {}
for r in rows:
    cat = r[-2]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

print(f"Generated {N} rows -> {OUTPUT_PATH}")
print(f"  Pass : {pass_count} ({pass_count/N*100:.1f}%)")
print(f"  Fail : {fail_count} ({fail_count/N*100:.1f}%)")
print("  Failure breakdown:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"    {cat:<22} {cnt:>5} ({cnt/N*100:.1f}%)")
