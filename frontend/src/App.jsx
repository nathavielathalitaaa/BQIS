/**
 * App — Root component
 * Sets up React Router with sidebar + scrollable content layout
 */
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar              from './components/Sidebar'
import Dashboard            from './pages/Dashboard'
import DataInput            from './pages/DataInput'
import SampleRiskOverview   from './pages/SampleRiskOverview'
import FailurePattern       from './pages/FailurePattern'
import ParameterImportance  from './pages/ParameterImportance'
import ExecutiveSummary     from './pages/ExecutiveSummary'

export default function App() {
  return (
    <BrowserRouter>
      <Sidebar />
      <main className="layout-content">
        <Routes>
          <Route path="/"                     element={<Dashboard />} />
          <Route path="/data-input"           element={<DataInput />} />
          <Route path="/sample-risk"          element={<SampleRiskOverview />} />
          <Route path="/failure-pattern"      element={<FailurePattern />} />
          <Route path="/parameter-importance" element={<ParameterImportance />} />
          <Route path="/executive-summary"    element={<ExecutiveSummary />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
