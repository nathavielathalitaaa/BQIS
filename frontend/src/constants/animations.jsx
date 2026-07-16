/**
 * Shared animation & loading constants for BQIS frontend.
 * Centralizes the fadeIn motion variant (previously duplicated across 5 page files)
 * and provides reusable skeleton loading components for professional UX.
 */

import React from 'react'
import PropTypes from 'prop-types'

// ─── Shared Framer Motion variant ─────────────────────────────────────────────
export const fadeIn = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.2 },
}

// ─── Skeleton Loading Components ──────────────────────────────────────────────
// Reusable shimmer placeholders shown while data is being fetched.

const shimmerStyle = {
  background: 'linear-gradient(90deg, #F0F0F5 25%, #E8E8EE 50%, #F0F0F5 75%)',
  backgroundSize: '200% 100%',
  animation: 'bqis-shimmer 1.4s ease-in-out infinite',
  borderRadius: 4,
}

const shimmerKeyframes = `
@keyframes bqis-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
`

export function SkeletonBlock({ height = 16, width = '100%', style = {} }) {
  return <div style={{ ...shimmerStyle, height, width, ...style }} />
}

SkeletonBlock.propTypes = {
  height: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  width: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  style: PropTypes.object,
}

SkeletonBlock.defaultProps = {
  height: 16,
  width: '100%',
  style: {},
}

export function SkeletonCard({ lines = 3, style = {} }) {
  return (
    <div className="clean-card" style={{ ...style }}>
      <div style={{ marginBottom: 14 }}>
        <SkeletonBlock height={18} width="40%" />
      </div>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <SkeletonBlock height={12} width={`${90 - i * 8}%`} />
        </div>
      ))}
    </div>
  )
}

SkeletonCard.propTypes = {
  lines: PropTypes.number,
  style: PropTypes.object,
}

SkeletonCard.defaultProps = {
  lines: 3,
  style: {},
}

export function SkeletonKpiRow({ count = 5 }) {
  return (
    <div className="grid-5 section-gap">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="kpi-card">
          <SkeletonBlock height={12} width="60%" style={{ marginBottom: 10 }} />
          <SkeletonBlock height={28} width="50%" style={{ marginBottom: 8 }} />
          <SkeletonBlock height={10} width="70%" />
        </div>
      ))}
    </div>
  )
}

SkeletonKpiRow.propTypes = {
  count: PropTypes.number,
}

SkeletonKpiRow.defaultProps = {
  count: 5,
}

export function SkeletonChart({ height = 300 }) {
  return (
    <div className="clean-card">
      <div style={{ marginBottom: 14 }}>
        <SkeletonBlock height={18} width="35%" />
      </div>
      <SkeletonBlock height={height} width="100%" />
    </div>
  )
}

SkeletonChart.propTypes = {
  height: PropTypes.number,
}

SkeletonChart.defaultProps = {
  height: 300,
}

export function SkeletonTable({ rows = 5, cols = 4 }) {
  return (
    <div className="clean-card">
      <div style={{ marginBottom: 14 }}>
        <SkeletonBlock height={18} width="30%" />
      </div>
      {Array.from({ length: rows + 1 }).map((_, r) => (
        <div key={r} style={{ display: 'flex', gap: 12, marginBottom: 10 }}>
          {Array.from({ length: cols }).map((_, c) => (
            <SkeletonBlock key={c} height={14} width={`${100 / cols}%`} />
          ))}
        </div>
      ))}
    </div>
  )
}

SkeletonTable.propTypes = {
  rows: PropTypes.number,
  cols: PropTypes.number,
}

SkeletonTable.defaultProps = {
  rows: 5,
  cols: 4,
}

export function LoadingSkeleton({ variant = 'default' }) {
  // Inject keyframes once
  if (typeof document !== 'undefined' && !document.getElementById('bqis-shimmer-style')) {
    const styleEl = document.createElement('style')
    styleEl.id = 'bqis-shimmer-style'
    styleEl.innerHTML = shimmerKeyframes
    document.head.appendChild(styleEl)
  }

  if (variant === 'dashboard') {
    return (
      <div style={{ padding: 32 }}>
        <SkeletonKpiRow count={5} />
        <div style={{ height: 16 }} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <SkeletonChart height={260} />
          <SkeletonChart height={260} />
        </div>
      </div>
    )
  }

  if (variant === 'risk') {
    return (
      <div style={{ padding: 32 }}>
        <SkeletonKpiRow count={5} />
        <div style={{ height: 16 }} />
        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 16 }}>
          <SkeletonChart height={300} />
          <SkeletonTable rows={4} cols={5} />
        </div>
      </div>
    )
  }

  if (variant === 'failure') {
    return (
      <div style={{ padding: 32 }}>
        <SkeletonKpiRow count={4} />
        <div style={{ height: 16 }} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16 }}>
          <SkeletonChart height={350} />
          <SkeletonCard lines={4} />
        </div>
      </div>
    )
  }

  if (variant === 'param') {
    return (
      <div style={{ padding: 32 }}>
        <SkeletonKpiRow count={4} />
        <div style={{ height: 16 }} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16 }}>
          <SkeletonChart height={390} />
          <SkeletonCard lines={5} />
        </div>
      </div>
    )
  }

  if (variant === 'exec') {
    return (
      <div style={{ padding: 32 }}>
        <SkeletonKpiRow count={4} />
        <div style={{ height: 16 }} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <SkeletonCard lines={5} />
          <SkeletonChart height={250} />
        </div>
      </div>
    )
  }

  // default fallback
  return (
    <div style={{ padding: 32, fontSize: '0.8rem', color: 'var(--c-text-2)' }}>Loading…</div>
  )
}

LoadingSkeleton.propTypes = {
  variant: PropTypes.oneOf(['default', 'dashboard', 'risk', 'failure', 'param', 'exec']),
}

LoadingSkeleton.defaultProps = {
  variant: 'default',
}