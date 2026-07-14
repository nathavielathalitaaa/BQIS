/**
 * Sidebar — exact replica of Streamlit app.py sidebar
 * Navy #00205B, white active state, footer with SNI reference
 */
import { NavLink, useLocation } from 'react-router-dom'
import {
  MdDashboard,
  MdBiotech,
  MdMap,
  MdBarChart,
  MdArticle,
} from 'react-icons/md'

const NAV_ITEMS = [
  { to: '/',                     icon: MdDashboard, label: 'Dashboard' },
  { to: '/sample-risk',          icon: MdBiotech,   label: 'Sample Risk Overview' },
  { to: '/failure-pattern',      icon: MdMap,       label: 'Failure Pattern Map' },
  { to: '/parameter-importance', icon: MdBarChart,  label: 'Parameter Importance' },
  { to: '/executive-summary',    icon: MdArticle,   label: 'Executive Summary' },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="layout-sidebar">
      {/* Logo */}
      <div className="sidebar-logo-area">
        <div className="sidebar-logo-title">BQIS</div>
        <div className="sidebar-logo-subtitle">TÜV NORD</div>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => {
          const isActive =
            to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)
          return (
            <NavLink
              key={to}
              to={to}
              className={`nav-item${isActive ? ' active' : ''}`}
              style={{ textDecoration: 'none' }}
            >
              <Icon className="nav-icon" />
              {label}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div style={{ fontWeight: 600, marginBottom: 2, color: 'rgba(255,255,255,0.55)' }}>
          AI Open Innovation 2026
        </div>
        <div>Case Provider: TÜV NORD Indonesia</div>
        <div style={{ marginTop: 2 }}>Standard: SNI 2973:2022</div>
      </div>
    </aside>
  )
}
