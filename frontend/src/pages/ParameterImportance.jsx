/**
 * ParameterImportance — Page 4
 * Streamlit render_param_importance() replica:
 * Filter strip, 4 KPIs, SHAP chart (2.8fr left) + AI Insight panel (1fr right), ranking table, bottom stats
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchShap } from '../services/api'
import Header    from '../components/Header'
import StatCard  from '../components/StatCard'
import FilterBar from '../components/FilterBar'
import ShapChart from '../components/ShapChart'
import DataTable from '../components/DataTable'

const fadeIn = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.2 } }

export default function ParameterImportance() {
  const [data, setData] = useState(null)
  const [period, setPeriod] = useState("All Time")
  const [batch, setBatch] = useState("All Batches")
  const [product, setProduct] = useState("All Products")
  const [isLoading, setIsLoading] = useState(false)

  const loadData = () => {
    setIsLoading(true)
    fetchShap()
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

  if (!data) return <div style={{ padding: 32, fontSize: '0.8rem', color: 'var(--c-text-2)' }}>Loading…</div>

  const { paramCount, affectedSamples, parameters } = data
  const sorted = [...parameters].sort((a, b) => b.meanAbs - a.meanAbs)
  const top = sorted[0]
  const mid = sorted[Math.floor(sorted.length / 2)]

  const tableCols = [
    {
      key: 'label', label: 'Parameter',
      render: (v, row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: `rgba(0,32,91,${Math.max(0.2, 1 - sorted.indexOf(row) * 0.08)})`,
            display: 'inline-block', flexShrink: 0,
          }} />
          <span style={{ fontWeight: 600 }}>{v}</span>
        </div>
      ),
    },
    {
      key: 'meanAbs', label: 'Mean (SHAP Value)', center: true,
      render: v => <strong>{v.toFixed(4)}</strong>,
    },
    {
      key: 'relativePct', label: 'Relative Influence (%)', center: true,
      render: (v) => (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
          <div style={{
            height: 5, width: Math.max(4, v * 2.8), borderRadius: 2,
            background: `rgba(0,32,91,${Math.max(0.2, v / 25)})`,
          }} />
          <span>{v.toFixed(1)}%</span>
        </div>
      ),
    },
    {
      key: 'direction', label: 'Impact Direction', center: true,
      render: v => (
        <span style={{ fontWeight: 600, color: v === 'Positive' ? '#E74C3C' : '#2ECC71' }}>
          {v === 'Positive' ? '↑ Positive' : '↓ Negative'}
        </span>
      ),
    },
    { key: 'interpretation', label: 'Interpretation' },
  ]

  return (
    <motion.div {...fadeIn}>
      <Header
        breadcrumbs={['Layer 3', 'Audit Intelligence Dashboard', 'Parameter Importance']}
        title="Parameter Importance Ranking (SHAP)"
        subtitle="Explanation of the laboratory parameters that contribute most to the XGBoost prediction using SHAP Explainability — Layer 3 Module"
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
            Updating parameter importance...
          </div>
        )}

        {/* 4 KPI Cards */}
        <div className="grid-4 section-gap">
          <StatCard label="Parameter Analyzed"    value={paramCount} desc="Total laboratory parameters" />
          <StatCard label="Top Ranked Parameters" value={top.label} desc="Highest SHAP value" accent="var(--c-fail)" />
          <StatCard label="Average SHAP Value"    value={top.meanAbs.toFixed(4)} desc="Mean absolute SHAP value" />
          <StatCard label="Affected Samples"      value={affectedSamples} desc="XGBoost prediction accuracy" />
        </div>

        {/* Main 2-column: SHAP chart (2.8fr) + AI Insight (1fr) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16 }} className="section-gap">

          {/* SHAP Chart card */}
          <div className="clean-card">
            <div className="sec-title">TOP Parameter Importance (SHAP)</div>
            <ShapChart data={parameters} height={390} />
            <div style={{
              marginTop: 10, padding: '8px 10px',
              background: '#F8F9FA', border: '1px solid var(--c-border)', borderRadius: 2,
              fontSize: '0.75rem', color: '#777', lineHeight: 1.6,
            }}>
              Displays the relative contribution of each laboratory parameter to the XGBoost prediction.
              Higher SHAP values indicate stronger influence on the prediction outcome.
            </div>
          </div>

          {/* AI Insight Panel — right_panel style */}
          <div>
            <div className="right-panel-wrap">
              <div className="right-panel-header">AI INSIGHT</div>
              <div className="right-panel-body">
                <div style={{ fontSize: '0.75rem', color: '#555', lineHeight: 1.6, marginBottom: 14 }}>
                  This analysis identifies the laboratory parameters that influence AI predictions
                  using SHAP explainability.
                </div>

                {/* Highest influence */}
                <div style={{
                  background: '#FDEDEC', border: '1px solid #F5B7B1',
                  borderRadius: 2, padding: '10px 12px', marginBottom: 10,
                }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--c-fail)', marginBottom: 3 }}>
                    {top.label} ({top.meanAbs.toFixed(4)})
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#555', lineHeight: 1.55 }}>
                    Significantly contributes to the prediction outcome. {top.relativePct.toFixed(1)}% relative influence.
                  </div>
                </div>

                {/* Moderate */}
                <div style={{
                  background: '#FEF9E7', border: '1px solid #F9E79F',
                  borderRadius: 2, padding: '10px 12px', marginBottom: 10,
                }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#D4AC0D', marginBottom: 3 }}>
                    Moderate Parameter
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#555', lineHeight: 1.55 }}>
                    {mid.label} ({mid.meanAbs.toFixed(4)}) — moderately influences the AI model decision.
                  </div>
                </div>

                {/* Low influence */}
                <div style={{
                  background: '#E8F8F5', border: '1px solid #A3E4D7',
                  borderRadius: 2, padding: '10px 12px',
                }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#117A65', marginBottom: 3 }}>
                    Low Influence
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#555', lineHeight: 1.55 }}>
                    Fat Content, Cadmium, and Acid Insoluble Ash have lower but still relevant contributions.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Ranking table */}
        <div className="clean-card section-gap">
          <div className="sec-title">Parameter Importance Ranking (SHAP)</div>
          <DataTable columns={tableCols} rows={sorted} />
        </div>

        {/* Bottom statistics */}
        <div className="clean-card section-gap">
          <div className="sec-title">Analysis Statistics</div>
          <div className="grid-4">
            {[
              { label: 'Total SHAP Mass',    value: sorted.reduce((s,p) => s+p.meanAbs, 0).toFixed(4), sub: 'Sum of all mean |SHAP| values' },
              { label: 'Top 3 Influence',    value: `${sorted.slice(0,3).reduce((s,p) => s+p.relativePct, 0).toFixed(1)}%`, sub: 'Accounted by top 3 parameters' },
              { label: 'Positive Direction', value: `${sorted.filter(p => p.direction === 'Positive').length}`, sub: 'Parameters increasing Fail probability' },
              { label: 'Negative Direction', value: `${sorted.filter(p => p.direction !== 'Positive').length}`, sub: 'Parameters decreasing Fail probability' },
            ].map((kpi, i) => (
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
