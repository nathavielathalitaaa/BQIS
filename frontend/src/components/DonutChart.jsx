/**
 * DonutChart — Streamlit go.Pie replica
 * Matches: hole=0.6, textinfo="percent+label", textfont size=12
 * Center label and custom legend matching Streamlit style
 */
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
} from 'recharts'
import PropTypes from 'prop-types'

const RADIAN = Math.PI / 180

const renderLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
  if (percent < 0.04) return null
  const r = innerRadius + (outerRadius - innerRadius) * 0.55
  const x = cx + r * Math.cos(-midAngle * RADIAN)
  const y = cy + r * Math.sin(-midAngle * RADIAN)
  return (
    <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central"
      style={{ fontSize: 11, fontFamily: 'Inter', fontWeight: 700 }}>
      {`${(percent * 100).toFixed(1)}%`}
    </text>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]
  return (
    <div style={{
      background: '#fff', border: '1px solid #E0E0E5', borderRadius: 2,
      padding: '7px 12px', fontSize: 12, boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
    }}>
      <div style={{ fontWeight: 700, color: d.payload.color, marginBottom: 2 }}>{d.name}</div>
      <div>Samples: <strong>{d.value.toLocaleString()}</strong></div>
    </div>
  )
}

export default function DonutChart({ data, centerLabel, centerSub }) {
  const total = data.reduce((s, d) => s + d.value, 0)
  return (
    <div style={{ width: '100%' }}>
      <div style={{ position: 'relative' }}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie
              data={data}
              cx="50%" cy="50%"
              innerRadius={78} outerRadius={125}
              dataKey="value"
              labelLine={false}
              label={renderLabel}
              strokeWidth={1}
              stroke="#fff"
            >
              {data.map((entry, i) => <Cell key={i} fill={entry.color} />)}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {/* Center text overlay */}
        {centerLabel && (
          <div style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center', pointerEvents: 'none',
          }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#111', lineHeight: 1 }}>
              {centerLabel}
            </div>
            {centerSub && (
              <div style={{ fontSize: 10, color: '#777', marginTop: 3 }}>{centerSub}</div>
            )}
          </div>
        )}
      </div>

      {/* Legend — matching Streamlit showlegend style */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '8px 18px',
        justifyContent: 'center', marginTop: 10,
      }}>
        {data.map((d, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: d.color, display: 'inline-block', flexShrink: 0 }} />
            <span style={{ color: '#555' }}>{d.name}</span>
            <strong style={{ color: '#111', marginLeft: 2 }}>{d.value.toLocaleString()}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}

DonutChart.propTypes = {
  data: PropTypes.arrayOf(PropTypes.shape({
    name: PropTypes.string.isRequired,
    value: PropTypes.number.isRequired,
    color: PropTypes.string.isRequired,
  })).isRequired,
  centerLabel: PropTypes.string,
  centerSub: PropTypes.string,
}

DonutChart.defaultProps = {
  centerLabel: '',
  centerSub: '',
}