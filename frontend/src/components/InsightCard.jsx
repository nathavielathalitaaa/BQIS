/**
 * InsightCard — AI Insight panel section item
 * Used in the right-side AI Insight panel on Parameter Importance page
 */
export default function InsightCard({ title, body, variant = 'default' }) {
  const styles = {
    default:  { bg: '#EFF6FF', border: '#BFDBFE', titleColor: '#1E40AF' },
    high:     { bg: '#FEF2F2', border: '#FECACA', titleColor: '#B91C1C' },
    medium:   { bg: '#FFFBEB', border: '#FDE68A', titleColor: '#92400E' },
    low:      { bg: '#F0FDF4', border: '#BBF7D0', titleColor: '#166534' },
    neutral:  { bg: '#F8F9FA', border: '#E0E0E5', titleColor: '#374151' },
  }
  const s = styles[variant] || styles.default

  return (
    <div style={{
      background: s.bg,
      border: `1px solid ${s.border}`,
      borderRadius: 3,
      padding: '10px 12px',
      marginBottom: 10,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: s.titleColor, marginBottom: 4 }}>
        {title}
      </div>
      <div style={{ fontSize: 11, color: 'var(--color-text-2)', lineHeight: 1.55 }}>
        {body}
      </div>
    </div>
  )
}
