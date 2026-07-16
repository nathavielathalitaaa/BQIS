import { useState, useEffect } from 'react'
import { fetchFilterOptions } from '../services/api'
import PropTypes from 'prop-types'

/**
 * FilterBar — Streamlit .filter-strip replication
 * Horizontal strip: FILTERS label | select | select | select | button
 * Each select has a small label above it (matching Streamlit st.selectbox label)
 */
export default function FilterBar({
  period,
  batch,
  product,
  onPeriodChange,
  onBatchChange,
  onProductChange,
  onApply,
  buttonLabel = 'Apply Filters'
}) {
  const [batchesList, setBatchesList] = useState([])
  const [productsList, setProductsList] = useState([])

  useEffect(() => {
    fetchFilterOptions()
      .then(res => {
        setBatchesList(res.data.batches || [])
        setProductsList(res.data.products || [])
      })
      .catch(err => {
        console.error("Failed to fetch filter options", err)
      })
  }, [])

  return (
    <div className="filter-strip">
      <span className="filter-strip-label">FILTERS</span>

      <div className="filter-select-wrap">
        <span className="filter-select-label">Analysis Period</span>
        <select
          className="filter-select"
          value={period}
          onChange={(e) => onPeriodChange(e.target.value)}
        >
          <option value="All Time">All Time</option>
          <option value="June 2026">June 2026</option>
          <option value="May 2026">May 2026</option>
          <option value="April 2026">April 2026</option>
        </select>
      </div>

      <div className="filter-select-wrap">
        <span className="filter-select-label">Laboratory Batch</span>
        <select
          className="filter-select"
          value={batch}
          onChange={(e) => onBatchChange(e.target.value)}
        >
          <option value="All Batches">All Batches</option>
          {batchesList.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
      </div>

      <div className="filter-select-wrap">
        <span className="filter-select-label">Product Category</span>
        <select
          className="filter-select"
          value={product}
          onChange={(e) => onProductChange(e.target.value)}
        >
          <option value="All Products">All Products</option>
          {productsList.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>

      <button className="btn-primary" onClick={onApply}>
        {buttonLabel}
      </button>
    </div>
  )
}

FilterBar.propTypes = {
  period: PropTypes.string.isRequired,
  batch: PropTypes.string.isRequired,
  product: PropTypes.string.isRequired,
  onPeriodChange: PropTypes.func.isRequired,
  onBatchChange: PropTypes.func.isRequired,
  onProductChange: PropTypes.func.isRequired,
  onApply: PropTypes.func.isRequired,
  buttonLabel: PropTypes.string,
}

FilterBar.defaultProps = {
  buttonLabel: 'Apply Filters',
}
