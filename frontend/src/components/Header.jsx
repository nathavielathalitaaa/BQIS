/**
 * Header — page breadcrumb + title + subtitle
 * Matches Streamlit: breadcrumb(0.75rem/muted), page-title(1.4rem/navy/700), page-subtitle(0.8rem/muted)
 */
export default function Header({ breadcrumbs = [], title, subtitle }) {
  return (
    <div className="page-header">
      {breadcrumbs.length > 0 && (
        <div className="breadcrumb">
          {breadcrumbs.map((c, i) => (
            <span key={i}>
              {c}
              {i < breadcrumbs.length - 1 && <span>›</span>}
            </span>
          ))}
        </div>
      )}
      <div className="page-title">{title}</div>
      {subtitle && <div className="page-subtitle">{subtitle}</div>}
    </div>
  )
}
