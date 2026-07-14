/**
 * RecommendationCard — Red-accented recommendation box
 * Used in Failure Pattern and Executive Summary pages
 */
export default function RecommendationCard({ title = 'Audit Recommendation', body }) {
  return (
    <div className="rec-box">
      <div className="rec-box-title">⚠ {title}</div>
      <div className="rec-box-body">{body}</div>
    </div>
  )
}
