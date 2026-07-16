/**
 * StatCard — KPI card
 * Matches Streamlit .kpi-card exactly:
 * label: 0.75rem / #555 / weight600
 * number: 1.8rem / #111 / weight700
 * sub: 0.7rem / #777
 */
import { motion } from 'framer-motion'
import PropTypes from 'prop-types'

export default function StatCard({ label, value, desc, accent }) {
  return (
    <motion.div
      className="kpi-card"
      whileHover={{ scale: 1.01 }}
      transition={{ duration: 0.15 }}
    >
      <div className="kpi-label">{label}</div>
      <div
        className="kpi-number"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
      {desc && <div className="kpi-sub">{desc}</div>}
    </motion.div>
  )
}

StatCard.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  desc: PropTypes.string,
  accent: PropTypes.string,
}

StatCard.defaultProps = {
  desc: '',
  accent: undefined,
}
