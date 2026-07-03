# BQIS - Experiment Log

## Approach 1: Baseline Clustering (Full-Feature)
**Status: CONCLUDED — diarsipkan sebagai baseline pembanding**

### Metodologi
- Subset data: hanya sampel `Historical_Status = Fail` (n=348)
- Fitur yang dipakai: **semua 11 parameter numerik sekaligus** (Moisture, Fat, Protein,
  Water_Activity, Acid_Insoluble_Ash, Acid_Value, Peroxide_Value, Total_Plate_Count,
  Yeast_Mold_Count, Lead_Pb, Cadmium_Cd) — tanpa seleksi fitur
- Preprocessing: KNN Imputation (n_neighbors=5) → StandardScaler
- Algoritma: K-Means (k dipilih via Silhouette Score) dan DBSCAN
- Validasi: Adjusted Rand Index (ARI) terhadap `Failure_Category` asli (ground truth
  dari proses generate dataset)

### Hasil
| Metode | Parameter | Hasil |
|---|---|---|
| K-Means | k optimal (silhouette) | k=6, silhouette=0.146 |
| K-Means | ARI vs Failure_Category | **0.130** (sangat lemah) |
| DBSCAN | eps=2.0, min_samples=5 | 35.6% sampel jadi noise (124/348) |
| DBSCAN | ARI vs Failure_Category | **0.281** (lemah, tapi lebih baik dari K-Means) |

### Interpretasi
Kedua algoritma menghasilkan keselarasan yang lemah terhadap kategori kegagalan asli.
DBSCAN sedikit lebih baik daripada K-Means, kemungkinan karena DBSCAN tidak memaksa
setiap sampel masuk ke suatu cluster (bisa jadi noise), sementara K-Means memaksa
partisi kaku.

### Dugaan Penyebab
1. **Fitur tidak diseleksi** — seluruh 11 parameter diperlakukan setara dalam
   perhitungan jarak, padahal secara teori tiap kategori kegagalan didominasi
   subset parameter yang berbeda (mis. Heavy_Metal hanya relevan dengan Pb/Cd,
   Microbiological hanya relevan dengan TPC/Yeast_Mold). Sinyal dari parameter
   relevan kemungkinan "tenggelam" oleh parameter yang tidak relevan untuk
   kategori tersebut.
2. **Kegagalan bersifat multi-faktor** — kemungkinan sejumlah sampel gagal karena
   lebih dari satu kategori sekaligus, sementara ground truth hanya mencatat SATU
   kategori dominan, menciptakan ambiguitas struktural pada label itu sendiri.

### Keputusan
Approach 1 diarsipkan sebagai baseline pembanding. Dilanjutkan ke **Approach 2**
dengan metodologi berbeda (feature selection / weighted feature space) untuk
menguji apakah dugaan penyebab #1 di atas benar.

### File Terkait (Approach 1)
- `BQIS_01_EDA_Preprocessing_Modeling.ipynb`
- `BQIS_02_Clustering_Failure_Pattern.ipynb`
- `bqis_clustering_result.csv`

---

## Approach 2: Feature-Selected Multi-Method Clustering
**Status: CONCLUDED**

### Metodologi
- Feature selection: Mutual Information (top-5 fitur, vs Approach 1 yang pakai 11 fitur mentah)
- 3 algoritma dibandingkan: K-Means, DBSCAN, Gaussian Mixture Model (GMM)
- Validasi: ARI + NMI (ditambah dari Approach 1 yang hanya pakai ARI)
- Tambahan: purity per kategori (bukan cuma skor global)

### Hasil
| Metode | ARI v1 | ARI v2 | NMI v2 |
|---|---|---|---|
| K-Means | 0.130 | 0.259 | 0.433 |
| DBSCAN | 0.281 | 0.239 | 0.218 |
| GMM | - | 0.101 | 0.130 |

### Purity per Kategori (K-Means, metode terpilih)
| Kategori | Purity |
|---|---|
| Stability | **1.00** (terpisah sempurna) |
| Microbiological | 0.54 |
| Heavy_Metal | 0.51 |
| Physicochemical | 0.42 |

### Interpretasi
Feature selection via Mutual Information terbukti signifikan meningkatkan performa
K-Means (ARI 2x lipat), mengonfirmasi dugaan Approach 1 bahwa sinyal tenggelam akibat
fitur tidak diseleksi. DBSCAN relatif stabil tapi menunjukkan sensitivitas tinggi
terhadap eps di ruang fitur yang lebih kecil (density landscape non-monoton).
GMM underperform, kemungkinan karena distribusi fitur lab (terutama Total_Plate_Count,
Yeast_Mold_Count) bersifat skewed dan tidak sesuai asumsi Gaussian simetris.

Kategori Stability menunjukkan pemisahan sempurna (purity=1.00), sementara
Physicochemical konsisten tumpang tindih di SEMUA metode (purity 0.42-0.59) —
mengindikasikan kegagalan physicochemical bersifat kontinu/multi-faktor secara
inheren, bukan artefak metode clustering tertentu.

### Keputusan
K-Means (feature-selected) ditetapkan sebagai metode clustering utama BQIS.

## Approach 3: Production-Ready Pipeline (Robustness Hardening)
**Status: CONCLUDED**

### Konteks
Approach 1-2 fokus mencari metode clustering terbaik. Approach 3 mengalihkan fokus ke
KESIAPAN pipeline menghadapi data baru (bukan dummy dataset ini), sesuai kekhawatiran
"bagaimana jika lolos ke tahap berikutnya dan diuji dengan data pabrik asli?"

### Yang Ditambahkan (sebelumnya TIDAK ADA di kode, meski diklaim di proposal Layer 1)
| Komponen | Fungsi | Status Sebelumnya |
|---|---|---|
| `COLUMN_MAP` | Mapping nama kolom, agar tidak hardcode ke skema dummy | Tidak ada |
| `flag_and_impute()` | Threshold missing >30% → flag manual review (klaim proposal) | Diklaim di dokumen, belum diimplementasi |
| `detect_outliers_iqr()` | Deteksi outlier ekstrem (IQR factor=3.0), FLAG bukan DIBUANG | Diklaim di dokumen ("outlier detection"), belum diimplementasi |
| `select_features()` | Fallback otomatis ke variance-based jika ground truth tidak tersedia | Tidak ada |
| `suggest_eps()` | eps DBSCAN otomatis via grid search + Silhouette (bukan trial-error manual) | Manual grid search |
| `check_sni_compliance()` | Hook cross-check ke baku mutu SNI 2973:2022 | Tidak ada |

### Keputusan Desain Penting
Outlier TIDAK dibuang otomatis — hanya di-flag untuk review manusia. Alasan: case
statement eksplisit menyebut "high production batch variation" sebagai limitation;
pembuangan otomatis berisiko menghapus variasi produksi asli, bukan sekadar error input.

### File Terkait
- `BQIS_04_Production_Ready_Pipeline.ipynb`
- `BQIS_04_v3_Robust.ipynb`

---

## Approach 4: Model Audit — Investigasi Leakage & Verifikasi SNI
**Status: CONCLUDED — TEMUAN KRITIS, WAJIB DIBACA SEBELUM PRESENTASI**

### Temuan 1: Data Leakage Parsial pada Classifier (XGBoost)
Ditemukan bahwa `generate_dummy_data.py` menetapkan `Historical_Status` melalui rule
IF-ELSE langsung dari parameter yang SAMA dipakai untuk training XGBoost (mis.
`moisture > 5.0 → Fail`). Ini berpotensi membuat akurasi classifier tinggi karena
menghafal aturan generator, bukan murni menemukan pola kompleks.

**Uji yang dilakukan:**
| Kondisi | Accuracy (5-fold CV) |
|---|---|
| Dengan seluruh fitur (termasuk rule-based) | 0.967 ± 0.006 |
| Tanpa 8 fitur rule-based | 0.775 |
| **Selisih** | **0.192 (19 poin persentase)** |

**Kesimpulan:** leakage bersifat PARSIAL, bukan total. Model tetap mempertahankan
performa jauh di atas baseline acak (~65%) tanpa fitur rule-based, mengindikasikan
sebagian sinyal asli tetap tertangkap. WAJIB dicantumkan sebagai catatan/disclaimer
setiap kali angka akurasi classifier dipresentasikan.

### Temuan 2: Validasi Held-Out Test (Train/Test Split Independen)
Ditambahkan test set 20% yang benar-benar terpisah dari proses training, CV, dan
feature selection (sebelumnya semua proses "melihat" data yang sama = double-dipping).

| Metrik | CV (train, 800 sampel) | Held-out test (200 sampel) |
|---|---|---|
| Accuracy | 0.9638 ± 0.0025 | 0.9800 |
| F1-Score | 0.9467 ± 0.0043 | 0.9710 |

**Kesimpulan:** selisih CV vs held-out kecil (1.6pp) → metodologi evaluasi TIDAK bias
terhadap proses seleksi fitur/model. Ini klaim metodologis yang independen dari
Temuan 1 (leakage) — keduanya menjawab pertanyaan berbeda dan harus dibedakan saat
presentasi.

### Temuan 3: Verifikasi SNI_LIMITS ke Dokumen Resmi (SNI 2973:2022, BSN)
Sebelumnya `SNI_LIMITS` berisi angka dari komentar `generate_dummy_data.py` (belum
terverifikasi). Dokumen resmi SNI 2973:2022 (Tabel 1) berhasil diperoleh dan
dikonfirmasi:

| Parameter | Generator Dummy | SNI 2973:2022 Resmi | Status |
|---|---|---|---|
| Kadar air | 5.0% | maks. 5% | Cocok |
| Abu tidak larut asam | 0.1% | maks. 0,1% | Cocok |
| **Bilangan asam** | **1.0** mg KOH/g | **maks. 2,0** mg KOH/g | **SALAH — dikoreksi** |
| Timbal (Pb) | 0.5 mg/kg | maks. 0,50 mg/kg | Cocok |
| Kadmium (Cd) | 0.2 mg/kg | maks. 0,20 mg/kg | Cocok |
| **Peroxide_Value** | 2.0 | **TIDAK ADA di SNI 2973:2022** | **Terkonfirmasi bukan parameter resmi** |

**Parameter SNI resmi yang BELUM ada di dataset (di luar scope PoC saat ini, bukan bug):**
Timah (Sn, maks 40 mg/kg), Merkuri (Hg, maks 0,05 mg/kg), Arsen (As, maks 0,50 mg/kg).
Cemaran mikroba (TPC, Kapang/Khamir) bersifat BERTINGKAT per jenis produk (Tabel 2-4
SNI) — dataset saat ini masih pakai satu angka generik, belum per-produk.

### Keputusan
1. Setiap presentasi angka akurasi classifier WAJIB menyertakan disclaimer soal data
   simulasi dan leakage parsial (Temuan 1).
2. `SNI_LIMITS` diupdate ke angka resmi terverifikasi (Temuan 3); parameter di luar
   scope didokumentasikan eksplisit sebagai keputusan scope, bukan kekurangan yang
   tidak disadari.
3. Simulator prediksi interaktif (`simulate_prediction()`) dibangun sebagai bukti
   sistem "hidup" untuk demo — SHAP explanation per-sampel divalidasi konsisten
   secara logis (mis. Moisture tinggi → SHAP mendorong FAIL, sesuai threshold SNI).

### File Terkait
- `BQIS_05_Model_Audit_Simulator.ipynb`

---

## Approach 5: Model Recalibration SOP
**Status: CONCLUDED**

### Konteks
Menjawab kekhawatiran: dataset dummy tidak akan pernah 100% merepresentasikan data
pabrik asli. Alih-alih mencoba "menyempurnakan" dummy data, fokus dialihkan ke
membangun PROTOKOL yang jelas untuk onboarding data baru — selaras dengan mindset
audit TÜV NORD sendiri (prosedur eksplisit, bukan janji performa semata).

### 5 Tahap SOP (meniru struktur SPC-TNI-020: Permohonan→Seleksi→Determinasi→Tinjauan→Keputusan)
1. **Data Intake & Mapping** — validasi kolom via `COLUMN_MAP`
2. **Quality Gate** — flag missing/outlier/duplikat (tidak buang outlier otomatis)
3. **Mandatory Retraining** — model classifier WAJIB dilatih ulang dari nol, TIDAK
   boleh mewarisi model yang dilatih dari dummy data
4. **Revalidation** — feature selection & clustering dihitung ulang, tidak mewarisi
   angka ARI/NMI dari dummy
5. **Human Sign-off & Model Card** — output JSON terstruktur, status default
   `PENDING_HUMAN_REVIEW`, field `approved_for_production` harus diubah manual oleh
   reviewer berwenang — model TIDAK otomatis dianggap siap pakai

### Hasil Uji Reproduksibilitas (dijalankan ulang di atas dataset dummy yang sama)
| Metrik | Approach 4 (manual) | Approach 5 (via SOP) | Konsisten? |
|---|---|---|---|
| Accuracy | 0.9670 ± 0.0060 | 0.9660 ± 0.0066 | Ya |
| ARI (K-Means) | 0.398 | 0.398 | Ya (persis sama) |
| NMI | 0.448 | 0.448 | Ya (persis sama) |

**Kesimpulan:** pipeline SOP terbukti reproducible — menghasilkan angka identik dengan
eksperimen manual sebelumnya, mengonfirmasi tidak ada bug/inkonsistensi dalam
otomatisasi Step 1-5.

### Temuan Tambahan
13.1% sampel (131/1000) ter-flag sebagai outlier ekstrem pada Quality Gate — belum
pernah terdeteksi di Approach 1-4 karena outlier detection baru diimplementasikan di
Approach 3. Perlu investigasi lanjut: apakah outlier ini terkonsentrasi di kategori
Physicochemical (yang sudah diketahui purity rendah).

### File Terkait
- `BQIS_06_Model_Recalibration_SOP.ipynb`

---

## Catatan Metodologis: ARI & NMI (untuk referensi tim)

**ARI (Adjusted Rand Index):** mengukur kesesuaian pengelompokan pada level PASANGAN
sampel, terkoreksi dari bias random assignment. Range -1 s.d. 1 (1=sempurna,
0=setara tebakan acak).

**NMI (Normalized Mutual Information):** mengukur besarnya informasi yang dibagi
antara hasil clustering dan kategori asli. Range 0 s.d. 1, lebih toleran terhadap
perbedaan jumlah cluster dibanding ARI.

Keduanya dipakai BERSAMAAN sebagai validasi silang — jika ARI dan NMI konsisten satu
sama lain, hasil clustering dianggap stabil, bukan artefak satu metrik saja. KEDUANYA
HANYA VALID jika ground truth (kategori asli) tersedia; untuk data tanpa sub-kategori
kegagalan (kemungkinan besar pada data pabrik asli), validasi fallback ke Silhouette
Score saja.

---

## Struktur Folder Terkini
```
BQIS/
├── data/
├── v1_eda_baseline_search/        (dulu: v1_baseline_full_features)
├── v2_feature_selection_search/   (dulu: v2_feature_selected_clustering)
├── v3_robust_pipeline/            (dulu: v3_production_ready — BUKAN eksperimen,
│                                    ini konsolidasi pipeline produksi)
├── v4_model_audit_validation/
└── v5_recalibration_sop/
```