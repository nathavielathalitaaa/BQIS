/**
 * SampleRiskOverview — Page 2
 * Streamlit-faithful: filter strip, 5 kpi cards, donut (left) + risk table (right), sample list
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchRiskOverview } from '../services/api'
import Header    from '../components/Header'
import StatCard  from '../components/StatCard'
import FilterBar from '../components/FilterBar'
import DonutChart from '../components/DonutChart'
import DataTable from '../components/DataTable'
import { fadeIn, LoadingSkeleton } from '../constants/animations'

const RISK_COLOR = {
  'Pass':        '#2ECC71',
  'High Risk':   '#E74C3C',
  'Medium Risk': '#F5B041',
  'Low Risk':    '#82E0AA',
}

export default function SampleRiskOverview() {
  const [data, setData] = useState(null)
  const [period, setPeriod] = useState("All Time")
  const [batch, setBatch] = useState("All Batches")
  const [product, setProduct] = useState("All Products")
  const [isLoading, setIsLoading] = useState(false)

  const loadData = () => {
    setIsLoading(true)
    fetchRiskOverview({ period, batch, product })
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

  if (!data) return <LoadingSkeleton variant="risk" />

  const { totalSamples, predictedPass, predictedFail, highRisk,
          avgConfidence, riskDistribution, riskTable, recentSamples } = data

  const highRiskBatches = Array.from(new Set(recentSamples
    .filter(s => s.risk === 'High Risk' || s.risk === 'High')
    .map(s => s.batch)
  )).filter(Boolean)

  const attentionMessage = highRiskBatches.length > 0
    ? `Suspend certification for batch${highRiskBatches.length > 1 ? 'es' : ''} ${highRiskBatches.join(' and ')} pending investigation.`
    : "Suspend certification for affected batches pending investigation."

  const riskCols = [
    {
      key: 'level', label: 'Risk Level',
      render: v => <span style={{ fontWeight: 700, color: RISK_COLOR[v] || '#111' }}>{v}</span>,
    },
    { key: 'count', label: 'Samples', center: true },
    { key: 'pct',   label: 'Percentage', center: true },
    { key: 'action', label: 'Recommended Action' },
    {
      key: 'priority', label: 'Priority', center: true,
      render: v => {
        if (v === '—') return <span style={{ color: '#bbb' }}>—</span>
        const cls = v === 'URGENT' ? 'badge badge-high' : v === 'HIGH' ? 'badge badge-medium' : 'badge badge-low'
        return <span className={cls}>{v}</span>
      },
    },
  ]

  const sampleCols = [
    {
      key: 'id', label: 'Sample ID',
      render: v => <span style={{ fontWeight: 600, color: 'var(--c-navy)' }}>{v}</span>,
    },
    { key: 'batch',   label: 'Batch Code' },
    { key: 'product', label: 'Product' },
    {
      key: 'prediction', label: 'Prediction', center: true,
      render: v => (
        <span style={{ fontWeight: 700, color: v === 'PASS' ? '#2ECC71' : '#E74C3C' }}>{v}</span>
      ),
    },
    {
      key: 'risk', label: 'Risk Level',
      render: v => <span style={{ color: RISK_COLOR[v] || '#111', fontWeight: 500 }}>{v}</span>,
    },
    {
      key: 'confidence', label: 'Confidence', center: true,
      render: v => `${v}%`,
    },
  ]

  return (
    <motion.div {...fadeIn}>
      <Header
        breadcrumbs={['Layer 3', 'Audit Intelligence Dashboard', 'Sample Risk Overview']}
        title="Sample Risk Overview"
        subtitle="Identify high-risk samples requiring priority inspection based on XGBoost prediction confidence scores."
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
            Updating risk overview...
          </div>
        )}

        {/* KPI row */}
        <div className="grid-5 section-gap">
          <StatCard label="Total Samples"    value={totalSamples.toLocaleString()} desc="Processed in selected period" />
          <StatCard label="Predicted PASS"   value={predictedPass.toLocaleString()} desc={`${((predictedPass/totalSamples)*100).toFixed(1)}% compliance rate`} accent="var(--c-pass)" />
          <StatCard label="Predicted FAIL"   value={predictedFail.toLocaleString()} desc={`${((predictedFail/totalSamples)*100).toFixed(1)}% non-compliance`} accent="var(--c-fail)" />
          <StatCard label="High Risk Samples" value={highRisk.toLocaleString()} desc="Requiring immediate review" accent="var(--c-fail)" />
          <StatCard label="Avg Confidence"   value={`${avgConfidence}%`} desc="Model prediction accuracy" />
        </div>

        {/* Donut + Risk Table row — matches Streamlit 2-column layout */}
        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 16 }} className="section-gap">

          {/* Left: Donut */}
          <div className="clean-card">
            <div className="sec-title">Risk Distribution</div>
            <DonutChart data={riskDistribution} centerLabel={totalSamples.toLocaleString()} centerSub="Samples" />
          </div>

          {/* Right: Table */}
          <div className="clean-card">
            <div className="sec-title">Risk Category Summary</div>
            <DataTable columns={riskCols} rows={riskTable} />

            <div style={{
              marginTop: 16, padding: '10px 14px',
              background: '#FDEDEC', border: '1px solid #F5B7B1', borderRadius: 2,
            }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--c-fail)', marginBottom: 4 }}>
                ⚠ Attention Required
              </div>
              <div style={{ fontSize: '0.72rem', color: '#7B241C', lineHeight: 1.6 }}>
                <strong>{highRisk}</strong> High Risk samples detected. Immediate auditor review is required.
                {" "}{attentionMessage}
              </div>
            </div>
          </div>
        </div>

        {/* Recent Samples Table */}
        <div className="clean-card section-gap">
          <div className="sec-title">Recent Sample Analysis</div>
          <DataTable columns={sampleCols} rows={recentSamples} />
        </div>
      </div>
    </motion.div>
  )
}
