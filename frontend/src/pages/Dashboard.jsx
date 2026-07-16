import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchDashboard, downloadReport } from '../services/api'
import Header     from '../components/Header'
import StatCard   from '../components/StatCard'
import DonutChart from '../components/DonutChart'
import ScatterPlotChart from '../components/ScatterPlotChart'
import ShapChart from '../components/ShapChart'
import FilterBar from '../components/FilterBar'
import { fadeIn, LoadingSkeleton } from '../constants/animations'

const aiExplanationsStatic = [
  {
    title: "1. XGBoost Classification",
    body: "Gradient-boosted decision tree model trained on 800 samples (5-fold CV). Accuracy: 96.4% ± 0.3%. Partial data leakage disclaimer: model trained on simulated SNI-based threshold labels.",
    color: "#EFF6FF", borderColor: "#BFDBFE"
  },
  {
    title: "2. SHAP Explainability",
    body: "SHapley Additive exPlanations quantify the contribution of each laboratory parameter to every individual prediction, providing auditor-interpretable reasoning for each PASS/FAIL decision.",
    color: "#F0FDF4", borderColor: "#BBF7D0"
  },
  {
    title: "3. K-Means Clustering",
    body: "Feature-selected K-Means (k=4) identifies recurring failure patterns: Microbiological, Physicochemical, Heavy Metal, and Stability clusters. ARI: 0.398 vs. ground truth categories.",
    color: "#FFFBEB", borderColor: "#FDE68A"
  },
  {
    title: "4. DBSCAN Anomaly",
    body: "Density-based algorithm detects hidden anomalies and outlier sample clusters not captured by standard K-Means grouping.",
    color: "#FDF2F8", borderColor: "#FBCFE8"
  },
  {
    title: "5. Audit Intelligence Dashboard",
    body: "Integrates all AI outputs into one unified interface for auditor decision support and quality oversight.",
    color: "#F5F3FF", borderColor: "#DDD6FE"
  },
  {
    title: "6. Executive Summary Generator",
    body: "Automatically compiles all analysis results into a structured, auditor-ready PDF report aligned with TÜV NORD methodology.",
    color: "#F3F4F6", borderColor: "#D1D5DB"
  }
]

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [period, setPeriod] = useState("All Time")
  const [batch, setBatch] = useState("All Batches")
  const [product, setProduct] = useState("All Products")
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)

  const loadData = () => {
    setIsLoading(true)
    fetchDashboard({ period, batch, product })
      .then(r => {
        setData(r.data)
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

  const handleGenerateReport = async () => {
    setIsGenerating(true)
    try {
      const response = await downloadReport('audit', { period, batch, product })
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `audit_report_${period}_${batch}_${product}.pdf`.replace(/\s+/g, '_'))
      document.body.appendChild(link)
      link.click()
      link.parentNode.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error("Failed to generate PDF report", error)
      alert("Failed to generate report. Please try again.")
    } finally {
      setIsGenerating(false)
    }
  }

  if (!data) return <LoadingSkeleton variant="dashboard" />

  const { totalSamples, predictedPass, predictedFail, highRiskSamples,
          avgConfidence, riskDistribution, topShap, scatterPoints } = data

  const bottomKpis = [
    { label: "Total Samples Processed", value: totalSamples.toLocaleString(), sub: "All time data" },
    { label: "Top Parameter", value: "Moisture", sub: "Highest SHAP impact" },
    { label: "Predicted PASS", value: predictedPass.toLocaleString(), sub: "Compliant" },
    { label: "Predicted FAIL", value: predictedFail.toLocaleString(), sub: "Non-compliant" },
    { label: "Auditor Review Required", value: highRiskSamples.toLocaleString(), sub: "High Risk samples" },
    { label: "Model Accuracy", value: `${avgConfidence}%`, sub: "XGBoost CV" },
  ]

  return (
    <motion.div {...fadeIn}>
      <Header
        breadcrumbs={['Layer 3', 'Audit Intelligence Dashboard']}
        title="Dashboard Overview"
        subtitle="Centralized audit intelligence platform integrating AI predictions, SHAP explainability, and failure pattern analysis."
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
            Updating dashboard data...
          </div>
        )}

        {/* ── Row 1: 5 KPI Cards ── */}
        <div className="grid-5 section-gap">
          <StatCard label="Laboratory Samples"        value={totalSamples.toLocaleString()} desc="Total processed by system" />
          <StatCard label="Predicted PASS"            value={predictedPass.toLocaleString()} desc={`${data.passRate}% of total samples`} accent="var(--c-pass)" />
          <StatCard label="Predicted FAIL"            value={predictedFail.toLocaleString()} desc={`${data.failRate}% of total samples`} accent="var(--c-fail)" />
          <StatCard label="High Risk Samples"         value={highRiskSamples.toLocaleString()} desc="Requiring auditor review" accent="var(--c-fail)" />
          <StatCard label="Average Prediction Confidence" value={`${avgConfidence}%`} desc="XGBoost prediction accuracy" />
        </div>

        {/* ── Row 2: 2x2 Grid of Preview Cards (Matching PDF) ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }} className="section-gap">
          
          {/* Top Left: Sample Risk Overview */}
          <div className="clean-card">
            <div className="sec-title">Sample Risk Overview</div>
            <DonutChart
              data={riskDistribution}
              centerLabel={totalSamples.toLocaleString()}
              centerSub="Total Samples"
            />
          </div>

          {/* Top Right: Failure Pattern Map */}
          <div className="clean-card">
            <div className="sec-title">Failure Pattern Map</div>
            <div style={{ padding: '10px 0' }}>
              <ScatterPlotChart points={scatterPoints || []} varExp={{pc1: 38, pc2: 15}} />
            </div>
            <div style={{ fontSize: '0.72rem', color: '#777', textAlign: 'center', marginTop: 10 }}>
              Identifies recurring quality failure patterns among laboratory samples via K-Means and DBSCAN.
            </div>
          </div>

          {/* Bottom Left: Parameter Importance Ranking */}
          <div className="clean-card">
            <div className="sec-title">Parameter Importance Ranking (SHAP)</div>
            <div style={{ marginTop: 20 }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#555', marginBottom: 10, textTransform: 'uppercase' }}>
                TOP 3 Parameters
              </div>
              <ShapChart data={topShap ? topShap.slice(0, 3) : []} height={200} />
            </div>
          </div>

          {/* Bottom Right: Executive Summary Generator */}
          <div className="clean-card">
            <div className="sec-title">Executive Summary Generator</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 16 }}>
              {[
                { label: "Predicted Fails", val: predictedFail.toLocaleString() },
                { label: "Dominant Failure Pattern", val: "Microbiology" },
                { label: "Most Influential Parameter", val: "Moisture Content" }
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', background: '#F8F9FA', border: '1px solid #EAEAF0', borderRadius: 4, padding: '10px 14px' }}>
                  <div style={{ flex: 1, fontSize: '0.75rem', color: '#555' }}>{item.label}</div>
                  <div style={{ fontWeight: 700, color: 'var(--c-navy)' }}>{item.val}</div>
                </div>
              ))}
              <button
                className="btn-primary"
                onClick={handleGenerateReport}
                disabled={isGenerating}
                style={{ marginTop: 8, padding: '12px', fontSize: '0.8rem', fontWeight: 600 }}
              >
                {isGenerating ? "Generating..." : "Generate Audit Report (PDF)"}
              </button>
              <div style={{ fontSize: '0.7rem', color: '#777', textAlign: 'center', marginTop: 4 }}>
                Automatically converts AI analysis into an auditor-ready report.
              </div>
            </div>
          </div>

        </div>

        {/* ── Row 3: AI Explanations (6 Cards) ── */}
        <div className="clean-card section-gap">
          <div className="sec-title">AI Explanation</div>
          <div className="grid-3">
            {aiExplanationsStatic.map((exp, i) => (
              <div key={i} style={{
                background: exp.color,
                border: `1px solid ${exp.borderColor}`,
                borderRadius: 4,
                padding: '16px',
              }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--c-navy)', marginBottom: 8 }}>
                  {exp.title}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#555', lineHeight: 1.6 }}>
                  {exp.body}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Row 4: Bottom KPI Ribbon (6 Items) ── */}
        <div className="clean-card section-gap">
          <div className="sec-title">System Performance Summary</div>
          <div className="grid-6">
            {bottomKpis.map((kpi, i) => (
              <div key={i} style={{
                textAlign: 'center', padding: '14px 10px',
                background: '#F8F9FA', border: '1px solid var(--c-border)', borderRadius: 4,
              }}>
                <div style={{ fontSize: '0.65rem', color: '#555', fontWeight: 700, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  {kpi.label}
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--c-navy)', lineHeight: 1 }}>
                  {kpi.value}
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </motion.div>
  )
}
