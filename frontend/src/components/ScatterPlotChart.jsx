/**
 * ScatterPlotChart — PCA cluster scatter plot
 * Each cluster shown in its designated color with a legend below
 * Colors imported from constants/colors.js (Palet A — standar resmi)
 */
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { FAILURE_COLORS, FAILURE_LABELS } from '../constants/colors'

// Aliases for local use
const CLUSTER_COLORS = FAILURE_COLORS
const CLUSTER_LABELS = FAILURE_LABELS

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #E0E0E5',
      borderRadius: 3,
      padding: '8px 12px',
      fontSize: 11,
      boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
    }}>
      <div style={{ fontWeight: 700, color: CLUSTER_COLORS[d.cluster], marginBottom: 4 }}>
        {CLUSTER_LABELS[d.cluster] || d.cluster}
      </div>
      <div>PC1: <strong>{d.pc1?.toFixed(3)}</strong></div>
      <div>PC2: <strong>{d.pc2?.toFixed(3)}</strong></div>
    </div>
  )
}

export default function ScatterPlotChart({ points, varExp }) {
  // Group points by cluster
  const clusters = {}
  points.forEach(p => {
    if (!clusters[p.cluster]) clusters[p.cluster] = []
    clusters[p.cluster].push({ pc1: p.pc1, pc2: p.pc2, cluster: p.cluster })
  })

  const pc1Label = `PC1 — Microbial / Chemical axis${varExp ? ` (${varExp.pc1}% variance)` : ''}`
  const pc2Label = `PC2 — Moisture / Physical axis${varExp ? ` (${varExp.pc2}% variance)` : ''}`

  return (
    <div>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F5" />
          <XAxis
            type="number"
            dataKey="pc1"
            name="PC1"
            tick={{ fontSize: 10, fontFamily: 'Inter', fill: '#888' }}
            tickLine={false}
            axisLine={{ stroke: '#E0E0E5' }}
            label={{ value: pc1Label, position: 'insideBottomRight', offset: -5, style: { fontSize: 10, fill: '#888', fontFamily: 'Inter' } }}
          />
          <YAxis
            type="number"
            dataKey="pc2"
            name="PC2"
            tick={{ fontSize: 10, fontFamily: 'Inter', fill: '#888' }}
            tickLine={false}
            axisLine={{ stroke: '#E0E0E5' }}
            label={{ value: pc2Label, angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#888', fontFamily: 'Inter' } }}
          />
          <ReferenceLine x={0} stroke="#E0E0E5" strokeDasharray="4 4" />
          <ReferenceLine y={0} stroke="#E0E0E5" strokeDasharray="4 4" />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />

          {Object.keys(clusters).map(key => (
            <Scatter
              key={key}
              name={CLUSTER_LABELS[key] || key}
              data={clusters[key]}
              fill={CLUSTER_COLORS[key] || '#999'}
              opacity={0.85}
              r={5}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', justifyContent: 'center', marginTop: 12 }}>
        {Object.keys(CLUSTER_LABELS).map(key => (
          <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
            <span style={{
              width: 10, height: 10, borderRadius: '50%',
              background: CLUSTER_COLORS[key], display: 'inline-block', flexShrink: 0,
            }} />
            <span style={{ color: 'var(--color-text-2)' }}>{CLUSTER_LABELS[key]}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
