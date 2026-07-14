/**
 * FailurePattern — Page 3
 * Streamlit render_failure_map() replica:
 * Filter strip, 4 KPIs, scatter plot (2.8fr) + pattern summary panel (1fr), methodology, bottom KPIs
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchClusters } from '../services/api'
import Header            from '../components/Header'
import StatCard          from '../components/StatCard'
import FilterBar         from '../components/FilterBar'
import ScatterPlotChart  from '../components/ScatterPlotChart'
import { FAILURE_COLORS } from '../constants/colors'

const fadeIn = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.2 } }

const RISK_BADGE = {
  High:   'badge badge-high',
  Medium: 'badge badge-medium',
  Low:    'badge badge-low',
}

// Warna dari Palet A (standar resmi via constants/colors.js)
const CLUSTER_METADATA = {
  Microbiological: {
    label: "Microbiology",
    color: FAILURE_COLORS.Microbiological,  // #E74C3C
    risk: "High",
    desc: "Samples exhibiting total plate count and coliform exceedances beyond SNI 2973:2022 microbiological limits. Most concentrated in recent batches.",
    recommendation: "Immediate re-inspection required. Suspend certification for affected batches pending hygiene investigation."
  },
  Physicochemical: {
    label: "Chemical",
    color: FAILURE_COLORS.Physicochemical,  // #3498DB
    risk: "High",
    desc: "Acid Value, Peroxide Value, or fat quality deviating from SNI thresholds — indicating lipid oxidation.",
    recommendation: "Halt certification of chemical-flagged samples. Trace raw material provenance for Lead and peroxide sources."
  },
  Heavy_Metal: {
    label: "Heavy Metal",
    color: FAILURE_COLORS.Heavy_Metal,      // #9B59B6
    risk: "High",
    desc: "Lead (Pb) or Cadmium (Cd) concentrations approaching or exceeding SNI limits.",
    recommendation: "Trace heavy metal contamination back to supplier. Mandatory supplier audit required."
  },
  Stability: {
    label: "Moisture",
    color: FAILURE_COLORS.Stability,        // #F5B041
    risk: "Medium",
    desc: "Samples with moisture content and water activity deviating beyond permissible thresholds, indicating process humidity control issues.",
    recommendation: "Selective re-testing of moisture-sensitive batches. Review drying and packaging line conditions."
  }
}

const staticBottomKpis = [
  { label: "K-Means ARI",         value: "0.398", sub: "vs. failure category ground truth" },
  { label: "NMI Score",           value: "0.448", sub: "Normalized Mutual Information" },
  { label: "Stability Purity",    value: "1.00",  sub: "Perfect cluster separation" },
  { label: "Microbiology Purity", value: "0.54",  sub: "K-Means cluster purity" },
  { label: "DBSCAN Noise",        value: "35.6%", sub: "Samples classified as noise" },
  { label: "Silhouette Score",    value: "0.259", sub: "K-Means feature-selected" }
]

export default function FailurePattern() {
  const [data, setData]                = useState(null)
  const [selectedCluster, setSelected] = useState(null)
  const [period, setPeriod]            = useState("All Time")
  const [batch, setBatch]              = useState("All Batches")
  const [product, setProduct]          = useState("All Products")
  const [isLoading, setIsLoading]      = useState(false)

  const loadData = () => {
    setIsLoading(true)
    fetchClusters({ period, batch, product })
      .then(r => {
        const enrichedProfiles = (r.data.clusterProfiles || []).map(p => {
          const meta = CLUSTER_METADATA[p.key] || {}
          return {
            ...p,
            ...meta,
            label: meta.label || p.key,
            color: meta.color || '#333',
            risk: meta.risk || 'Unknown',
            desc: meta.desc || '',
            recommendation: meta.recommendation || ''
          }
        })
        
        const enrichedData = {
          ...r.data,
          clusterProfiles: enrichedProfiles,
          bottomKpis: r.data.bottomKpis || staticBottomKpis
        }
        
        setData(enrichedData)
        
        if (selectedCluster) {
          const matched = enrichedProfiles.find(p => p.key === selectedCluster.key)
          setSelected(matched || enrichedProfiles[0] || null)
        } else {
          setSelected(enrichedProfiles[0] || null)
        }
        
        setIsLoading(false)
      })
      .catch(err => {
        console.error(err)
        setIsLoading(false)
      })
  }

  useEffect(() => {
    loadData()
  }, [])

  if (!data) return <div style={{ padding: 32, fontSize: '0.8rem', color: 'var(--c-text-2)' }}>Loading…</div>

  const { totalClusters, dominantCluster, highRiskClusters, affectedSamples,
          clusterProfiles, scatterPoints, varExp, bottomKpis } = data

  return (
    <motion.div {...fadeIn}>
      <Header
        breadcrumbs={['Layer 3', 'Audit Intelligence Dashboard', 'Failure Pattern Map']}
        title="Failure Pattern Map"
        subtitle="Recurring quality failure patterns identified via K-Means and DBSCAN clustering — Layer 3 Module"
      />

      <div className="content-inner">
        <FilterBar
          period={period}
          batch={batch}
          product={product}
          onPeriodChange={setPeriod}
          onBatchChange={setBatch}
          onProductChange={setProduct}
          onApply={loadData}
        />
        {isLoading && (
          <div style={{ padding: '4px 0', fontSize: '0.75rem', color: 'var(--c-navy)', fontStyle: 'italic', marginBottom: 12 }}>
            Updating failure patterns...
          </div>
        )}

        {/* 4 KPI Cards */}
        <div className="grid-4 section-gap">
          <StatCard label="Total Clusters"     value={totalClusters} desc="K-Means identified clusters" />
          <StatCard label="Dominant Pattern"   value={CLUSTER_METADATA[dominantCluster]?.label || dominantCluster} desc="Highest sample count cluster" />
          <StatCard label="High Risk Clusters" value={highRiskClusters} desc="Microbiology + Chemical" accent="var(--c-fail)" />
          <StatCard label="Affected Samples"   value={affectedSamples.toLocaleString()} desc="Across all failure clusters" accent="var(--c-medium)" />
        </div>

        {/* Main layout: 2.8fr scatter | 1fr right panel (matches Streamlit col_main, col_right) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16 }} className="section-gap">

          {/* ── LEFT: Scatter + Methodology ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Scatter Plot card */}
            <div className="clean-card">
              <div className="sec-title">Cluster Scatter Plot — Principal Component Analysis (PCA)</div>
              <ScatterPlotChart points={scatterPoints} varExp={varExp} />
            </div>

            {/* Methodology card */}
            <div className="clean-card">
              <div className="sec-title">Pattern Detection Methodology</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ background: '#EFF6FF', border: '1px solid #BFDBFE', borderRadius: 2, padding: '12px 14px' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#1E40AF', marginBottom: 6 }}>
                    K-Means Clustering
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#555', lineHeight: 1.65 }}>
                    Partitions laboratory samples into K=4 predefined clusters using feature-selected
                    Mutual Information top-5 parameters. ARI: 0.398. Stability cluster purity=1.00 (perfect separation).
                  </div>
                </div>
                <div style={{ background: '#FFFBEB', border: '1px solid #FDE68A', borderRadius: 2, padding: '12px 14px' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#92400E', marginBottom: 6 }}>
                    DBSCAN Anomaly Detection
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#555', lineHeight: 1.65 }}>
                    Identifies outlier samples not conforming to any core cluster pattern.
                    35.6% of samples classified as noise (eps auto-tuned via Silhouette grid search).
                    ARI: 0.239, NMI: 0.218 on feature-selected space.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ── RIGHT: Pattern Summary Panel (Streamlit right_panel) ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Cluster list */}
            <div className="right-panel-wrap">
              <div className="right-panel-header">Pattern Summary</div>
              <div className="right-panel-body" style={{ padding: 10 }}>
                {clusterProfiles.map((c, i) => (
                  <div
                    key={i}
                    className="cluster-row"
                    style={{
                      borderLeft: selectedCluster?.key === c.key
                        ? `3px solid ${c.color}`
                        : '3px solid transparent',
                    }}
                    onClick={() => setSelected(c)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: c.color, display: 'inline-block',
                      }} />
                      <span style={{ fontSize: '0.78rem', fontWeight: 600, color: c.color }}>
                        ● <span style={{ color: '#333' }}>{c.label}</span>
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span className={RISK_BADGE[c.risk]}>{c.risk}</span>
                      <span style={{ fontWeight: 700, color: '#555', fontSize: '0.8rem' }}>{c.samples}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Selected cluster profile */}
            {selectedCluster && (
              <div className="right-panel-wrap">
                <div className="right-panel-header" style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  textTransform: 'uppercase',
                }}>
                  <span>{selectedCluster.label}</span>
                  <span style={{
                    fontSize: '0.6rem', background: 'rgba(255,255,255,0.2)',
                    padding: '2px 5px', borderRadius: 2,
                  }}>
                    {selectedCluster.risk} Risk
                  </span>
                </div>
                <div className="right-panel-body">
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: 4 }}>Cluster Profile</div>
                  <div style={{ fontSize: '0.75rem', color: '#555', lineHeight: 1.6, marginBottom: 12 }}>
                    {selectedCluster.desc}
                  </div>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    borderTop: '1px solid var(--c-border)', paddingTop: 10,
                    textAlign: 'center',
                  }}>
                    <div>
                      <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: 2 }}>Samples</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#111' }}>{selectedCluster.samples}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: 2 }}>Risk</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: selectedCluster.color }}>
                        {selectedCluster.risk}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Audit Recommendation */}
            {selectedCluster && (
              <div className="rec-box">
                <div className="rec-box-title">Audit Recommendation</div>
                <div className="rec-box-body">{selectedCluster.recommendation}</div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom KPI metrics */}
        <div className="clean-card section-gap">
          <div className="sec-title">Clustering Performance Metrics</div>
          <div className="grid-6">
            {bottomKpis.map((kpi, i) => (
              <div key={i} style={{
                textAlign: 'center', padding: '12px 8px',
                background: '#F8F9FA', border: '1px solid var(--c-border)', borderRadius: 2,
              }}>
                <div style={{ fontSize: '0.68rem', color: '#555', fontWeight: 600, marginBottom: 5, textTransform: 'uppercase', letterSpacing: '0.3px' }}>
                  {kpi.label}
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--c-navy)', lineHeight: 1 }}>
                  {kpi.value}
                </div>
                <div style={{ fontSize: '0.65rem', color: '#777', marginTop: 4 }}>{kpi.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
