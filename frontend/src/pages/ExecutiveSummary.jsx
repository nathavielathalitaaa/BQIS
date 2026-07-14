/**
 * ExecutiveSummary — Page 5
 * Streamlit render_exec_summary() replica:
 * Filter strip (with Generate button), 4 KPIs, risk summary + mini SHAP, recommendation, generate card, bottom stats
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { MdFileDownload } from 'react-icons/md'
import { fetchExecutiveSummary, downloadReport } from '../services/api'
import Header    from '../components/Header'
import StatCard  from '../components/StatCard'
import FilterBar from '../components/FilterBar'
import ShapChart from '../components/ShapChart'

const fadeIn = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.2 } }

const RISK_COLORS = { High: '#E74C3C', Medium: '#F5B041', Low: '#2ECC71' }
const RISK_BG     = { High: '#FDEDEC', Medium: '#FEF9E7', Low: '#EAFAF1' }
const RISK_BORDER = { High: '#F5B7B1', Medium: '#F9E79F', Low: '#A9DFBF' }

export default function ExecutiveSummary() {
  const [data, setData] = useState(null)
  const [period, setPeriod] = useState("All Time")
  const [batch, setBatch] = useState("All Batches")
  const [product, setProduct] = useState("All Products")
  const [isLoading, setIsLoading] = useState(false)
  const [isGeneratingExec, setIsGeneratingExec] = useState(false)
  const [isGeneratingAudit, setIsGeneratingAudit] = useState(false)

  const loadData = () => {
    setIsLoading(true)
    fetchExecutiveSummary({ period, batch, product })
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

  const handleGenerateExecReport = async () => {
    setIsGeneratingExec(true)
    try {
      const response = await downloadReport('executive', { period, batch, product })
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `executive_summary_${period}_${batch}_${product}.pdf`.replace(/\s+/g, '_'))
      document.body.appendChild(link)
      link.click()
      link.parentNode.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error("Failed to generate Executive Summary PDF", error)
      alert("Failed to generate report. Please try again.")
    } finally {
      setIsGeneratingExec(false)
    }
  }

  const handleGenerateAuditReport = async () => {
    setIsGeneratingAudit(true)
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
      console.error("Failed to generate Audit Report PDF", error)
      alert("Failed to generate report. Please try again.")
    } finally {
      setIsGeneratingAudit(false)
    }
  }

  if (!data) return <div style={{ padding: 32, fontSize: '0.8rem', color: 'var(--c-text-2)' }}>Loading…</div>

  const { totalSamples, predictedPass, predictedFail, avgConfidence,
          passRate, failRate, riskSummary, topRisks, parameterImpact, auditRecommendation, bottomStats } = data

  const derivedBottomStats = [
    { label: "Samples Analyzed",   value: totalSamples.toLocaleString(), sub: "Selected period" },
    { label: "Compliance Rate",    value: `${passRate}%`, sub: "Predicted PASS" },
    { label: "Critical Failures",  value: (riskSummary?.high || 0).toLocaleString(), sub: "High risk — urgent" },
    { label: "Model Confidence",   value: `${avgConfidence}%`, sub: "Avg. XGBoost score" },
    { label: "Top Risk Parameter", value: parameterImpact?.[0]?.parameter || "Moisture", sub: `SHAP value` },
    { label: "Batch Flag",         value: batch && batch !== "All Batches" ? batch : "Multiple Batches", sub: "Active batch filter" }
  ]
  const bottomStatsToUse = bottomStats || derivedBottomStats

  return (
    <motion.div {...fadeIn}>
      <Header
        breadcrumbs={['Layer 3', 'Audit Intelligence Dashboard', 'Executive Summary']}
        title="Executive Summary Generator"
        subtitle="Automatically compile AI analysis into an audit-ready executive report."
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
          buttonLabel="Apply Filters"
        />
        {isLoading && (
          <div style={{ padding: '4px 0', fontSize: '0.75rem', color: 'var(--c-navy)', fontStyle: 'italic', marginBottom: 12 }}>
            Updating executive summary...
          </div>
        )}

        {/* 4 KPI Cards */}
        <div className="grid-4 section-gap">
          <StatCard label="Total Laboratory Samples"  value={totalSamples.toLocaleString()} desc={`${period} — ${product}`} />
          <StatCard label="Predicted PASS"            value={predictedPass.toLocaleString()} desc={`${passRate}%`} accent="var(--c-pass)" />
          <StatCard label="Predicted Fail"            value={predictedFail.toLocaleString()} desc={`${failRate}%`} accent="var(--c-fail)" />
          <StatCard label="Average Prediction Confidence" value={`${avgConfidence}%`} desc="XGBoost prediction accuracy" />
        </div>

        {/* Risk Summary (left) + Parameter Impact SHAP (right) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }} className="section-gap">

          {/* Risk Summary */}
          <div className="clean-card">
            <div className="sec-title">Risk Summary</div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
              {[
                { label: 'High Risk',   value: riskSummary.high,   key: 'High' },
                { label: 'Medium Risk', value: riskSummary.medium, key: 'Medium' },
                { label: 'Low Risk',    value: riskSummary.low,    key: 'Low' },
                { label: 'PASS',        value: riskSummary.pass,   key: null },
              ].map((r, i) => {
                const isPass = r.key === null
                return (
                  <div key={i} style={{
                    background: isPass ? '#EAF8F0' : RISK_BG[r.key],
                    border: `1px solid ${isPass ? '#A9DFBF' : RISK_BORDER[r.key]}`,
                    borderRadius: 2, padding: '10px 12px',
                  }}>
                    <div style={{ fontSize: '0.7rem', fontWeight: 700, color: isPass ? '#1E8449' : RISK_COLORS[r.key], textTransform: 'uppercase', marginBottom: 3, letterSpacing: '0.5px' }}>
                      {r.label}
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: isPass ? '#1E8449' : RISK_COLORS[r.key], lineHeight: 1 }}>
                      {r.value.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#777', marginTop: 2 }}>
                      {((r.value / totalSamples) * 100).toFixed(1)}% of total
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Top risks table */}
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#333', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.3px' }}>
              Top Risk Categories
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
              <thead style={{ background: '#F8F9FA', borderBottom: '2px solid var(--c-border)' }}>
                <tr>
                  {['Category','Samples','Risk','Action'].map((h, i) => (
                    <th key={i} style={{ padding: '7px 10px', textAlign: i === 1 ? 'center' : 'left', fontSize: '0.7rem', fontWeight: 700, color: '#555', textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {topRisks.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #F0F0F5', background: i % 2 === 1 ? '#FAFAFA' : '#fff' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 500 }}>{r.category}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center', fontWeight: 700, color: RISK_COLORS[r.risk] }}>{r.samples}</td>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{
                        fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px', borderRadius: 10,
                        background: RISK_BG[r.risk], border: `1px solid ${RISK_BORDER[r.risk]}`, color: RISK_COLORS[r.risk],
                      }}>{r.risk}</span>
                    </td>
                    <td style={{ padding: '7px 10px', color: '#555' }}>{r.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Parameter Impact mini SHAP */}
          <div className="clean-card">
            <div className="sec-title">Parameter Impact (Top 5 SHAP)</div>
            <ShapChart
              data={parameterImpact.map(p => ({ ...p, label: p.parameter, meanAbs: p.shapVal, relativePct: 0 }))}
              height={250}
            />
            <div style={{
              marginTop: 12, background: '#F8F9FA', border: '1px solid var(--c-border)', borderRadius: 2, padding: '10px 12px',
            }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--c-navy)', marginBottom: 4 }}>Key Findings</div>
              <div style={{ fontSize: '0.72rem', color: '#555', lineHeight: 1.6 }}>
                Moisture Content and Water Activity collectively account for &gt;40% of prediction influence.
                These parameters are the primary drivers of quality non-compliance under SNI 2973:2022.
              </div>
            </div>
          </div>
        </div>

        {/* Audit Recommendation */}
        <div className="clean-card section-gap">
          <div className="sec-title">AI-Generated Audit Recommendation</div>
          <div style={{
            background: '#FDEDEC', border: '1px solid #F5B7B1', borderRadius: 2, padding: '14px 18px',
          }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--c-fail)', marginBottom: 6 }}>
              ⚠ Audit Action Required — Batch {batch}
            </div>
            <div style={{ fontSize: '0.78rem', color: '#7B241C', lineHeight: 1.75 }}>
              {auditRecommendation}
            </div>
          </div>
        </div>

        {/* Generate Report + Insight panel */}
        <div style={{ textAlign: 'center', margin: '0 0 20px' }}>
          <div className="clean-card" style={{ display: 'inline-block', padding: '24px 32px', textAlign: 'center' }}>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--c-navy)', marginBottom: 6 }}>
              Generate Audit Reports
            </div>
            <div style={{ fontSize: '0.78rem', color: '#555', marginBottom: 16, lineHeight: 1.6 }}>
              Automatically convert AI analysis into auditor-ready PDF reports.
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
              <button
                className="btn-primary"
                onClick={handleGenerateExecReport}
                disabled={isGeneratingExec}
                style={{ padding: '10px 20px', fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: 8 }}
              >
                <MdFileDownload size={16} />
                {isGeneratingExec ? "Generating..." : "Generate Executive Summary Report (PDF)"}
              </button>
              <button
                className="btn-primary"
                onClick={handleGenerateAuditReport}
                disabled={isGeneratingAudit}
                style={{ padding: '10px 20px', fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: 8 }}
              >
                <MdFileDownload size={16} />
                {isGeneratingAudit ? "Generating..." : "Generate Audit Report (PDF)"}
              </button>
            </div>
          </div>
        </div>

        {/* Bottom Stats */}
        <div className="clean-card section-gap">
          <div className="sec-title">Summary Statistics — {period} · {batch} · {product}</div>
          <div className="grid-6">
            {bottomStatsToUse.map((kpi, i) => (
              <div key={i} style={{
                textAlign: 'center', padding: '12px 8px',
                background: '#F8F9FA', border: '1px solid var(--c-border)', borderRadius: 2,
              }}>
                <div style={{ fontSize: '0.68rem', color: '#555', fontWeight: 600, marginBottom: 5, textTransform: 'uppercase', letterSpacing: '0.3px' }}>
                  {kpi.label}
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--c-navy)', lineHeight: 1 }}>{kpi.value}</div>
                <div style={{ fontSize: '0.65rem', color: '#777', marginTop: 4 }}>{kpi.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
