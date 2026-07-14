/**
 * BQIS — Single Source of Truth for Category & Risk Colors
 * ---------------------------------------------------------
 * Palet A (standar resmi — dipakai di backend, Dashboard, ExecutiveSummary, SampleRiskOverview)
 * Semua komponen HARUS import dari sini. Jangan hardcode warna kategori di mana pun.
 */

// ── Failure Category Colors (4 kategori + Pass) ──────────────────────────────
export const FAILURE_COLORS = {
  Microbiological: '#E74C3C',
  Physicochemical: '#3498DB',
  Heavy_Metal:     '#9B59B6',
  Stability:       '#F5B041',
  Pass:            '#2ECC71',
}

// Label tampil untuk tiap kategori
export const FAILURE_LABELS = {
  Microbiological: 'Microbiology',
  Physicochemical: 'Chemical',
  Heavy_Metal:     'Heavy Metal',
  Stability:       'Moisture / Stability',
  Pass:            'Pass',
}

// ── Risk Level Colors ─────────────────────────────────────────────────────────
export const RISK_COLORS = {
  'High Risk':   '#E74C3C',
  'Medium Risk': '#F5B041',
  'Low Risk':    '#82E0AA',
  'Pass':        '#2ECC71',
  // Alias untuk key pendek yang dipakai di topRisks
  'High':        '#E74C3C',
  'Medium':      '#F5B041',
  'Low':         '#82E0AA',
}

// Background pastel (untuk badge / card background)
export const RISK_BG = {
  'High Risk':   '#FDEDEC',
  'Medium Risk': '#FEF9E7',
  'Low Risk':    '#E8F8F5',
  'Pass':        '#EAF8F0',
  'High':        '#FDEDEC',
  'Medium':      '#FEF9E7',
  'Low':         '#E8F8F5',
}

// Border pastel
export const RISK_BORDER = {
  'High Risk':   '#F5B7B1',
  'Medium Risk': '#F9E79F',
  'Low Risk':    '#A3E4D7',
  'Pass':        '#A9DFBF',
  'High':        '#F5B7B1',
  'Medium':      '#F9E79F',
  'Low':         '#A3E4D7',
}

// ── PDF / Report colors (hex strings untuk FPDF) ──────────────────────────────
// Diexport sebagai RGB tuple strings untuk sinkronisasi dengan backend
export const PDF_COLORS = {
  navy:             [0, 32, 91],
  white:            [255, 255, 255],
  lightGray:        [248, 249, 250],
  borderGray:       [224, 224, 229],
  highRisk:         [231, 76, 60],   // #E74C3C
  mediumRisk:       [245, 176, 65],  // #F5B041
  lowRisk:          [130, 224, 170], // #82E0AA
  pass:             [46, 204, 113],  // #2ECC71
  microbiological:  [231, 76, 60],   // #E74C3C
  physicochemical:  [52, 152, 219],  // #3498DB
  heavyMetal:       [155, 89, 182],  // #9B59B6
  stability:        [245, 176, 65],  // #F5B041
}
