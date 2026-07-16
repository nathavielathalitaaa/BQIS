/**
 * ShapChart — Streamlit go.Bar horizontal replica
 * Matches: orientation="h", marker color=#00205B, bargap=0.4, autorange="reversed"
 * Blue gradient bars, value labels on right, correct axis labels
 */
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, LabelList,
} from 'recharts'
import PropTypes from 'prop-types'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]
  return (
    <div style={{
      background: '#fff', border: '1px solid #E0E0E5', borderRadius: 2,
      padding: '7px 12px', fontSize: 12, boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
    }}>
      <div style={{ fontWeight: 700, color: '#00205B', marginBottom: 2 }}>{d.payload.label}</div>
      <div>SHAP Value: <strong>{d.value.toFixed(4)}</strong></div>
      {d.payload.relativePct > 0 && (
        <div>Relative: <strong>{d.payload.relativePct.toFixed(1)}%</strong></div>
      )}
    </div>
  )
}

export default function ShapChart({ data, height = 380 }) {
  const sorted = [...data].sort((a, b) => b.meanAbs - a.meanAbs)
  const maxVal = sorted[0]?.meanAbs || 1

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={sorted}
        layout="vertical"
        margin={{ top: 4, right: 55, bottom: 24, left: 90 }}
        barCategoryGap="30%"
      >
        <XAxis
          type="number"
          tick={{ fontSize: 9, fontFamily: 'Inter', fill: '#888' }}
          tickLine={false}
          axisLine={{ stroke: '#E0E0E5' }}
          tickFormatter={v => v.toFixed(3)}
          label={{
            value: 'Mean |SHAP| Value',
            position: 'insideBottomRight',
            offset: -8,
            style: { fontSize: 9, fill: '#888', fontFamily: 'Inter' },
          }}
        />
        <YAxis
          type="category"
          dataKey="label"
          tick={{ fontSize: 10, fontFamily: 'Inter', fill: '#444' }}
          tickLine={false}
          axisLine={false}
          width={85}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,32,91,0.04)' }} />
        <Bar dataKey="meanAbs" radius={[0, 2, 2, 0]}>
          {sorted.map((entry, i) => (
            <Cell key={i} fill={`rgba(0,32,91,${Math.max(0.3, 1 - i * 0.07)})`} />
          ))}
          <LabelList
            dataKey="meanAbs"
            position="right"
            formatter={v => v.toFixed(3)}
            style={{ fontSize: 9, fill: '#666', fontFamily: 'Inter' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

ShapChart.propTypes = {
  data: PropTypes.arrayOf(PropTypes.shape({
    label: PropTypes.string.isRequired,
    meanAbs: PropTypes.number.isRequired,
    relativePct: PropTypes.number,
  })).isRequired,
  height: PropTypes.number,
}

ShapChart.defaultProps = {
  height: 380,
}