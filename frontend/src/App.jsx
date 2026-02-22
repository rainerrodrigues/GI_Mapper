import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import ClusteringPage from './pages/ClusteringPage'
import ROIPredictionPage from './pages/ROIPredictionPage'
import AnomalyDetectionPage from './pages/AnomalyDetectionPage'
import MLAScoringPage from './pages/MLAScoringPage'
import ForecastingPage from './pages/ForecastingPage'
import RiskAssessmentPage from './pages/RiskAssessmentPage'
import PerformanceMonitoringPage from './pages/PerformanceMonitoringPage'
import './App.css'

function Navigation() {
  const location = useLocation()
  const [isOpen, setIsOpen] = useState(true)

  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/clustering', label: 'Clustering', icon: '🗺️' },
    { path: '/roi-prediction', label: 'ROI Prediction', icon: '💰' },
    { path: '/anomaly-detection', label: 'Anomaly Detection', icon: '🔍' },
    { path: '/mla-scoring', label: 'MLA Scoring', icon: '📈' },
    { path: '/forecasting', label: 'Forecasting', icon: '📉' },
    { path: '/risk-assessment', label: 'Risk Assessment', icon: '⚠️' },
    { path: '/monitoring', label: 'Monitoring', icon: '📡' },
  ]

  return (
    <nav className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-header">
        <h2>🌍 AI GIS Platform</h2>
        <button className="toggle-btn" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? '◀' : '▶'}
        </button>
      </div>
      <ul className="nav-list">
        {navItems.map((item) => (
          <li key={item.path}>
            <Link
              to={item.path}
              className={location.pathname === item.path ? 'active' : ''}
            >
              <span className="icon">{item.icon}</span>
              {isOpen && <span className="label">{item.label}</span>}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  )
}

function App() {
  return (
    <Router>
      <div className="app">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/clustering" element={<ClusteringPage />} />
            <Route path="/roi-prediction" element={<ROIPredictionPage />} />
            <Route path="/anomaly-detection" element={<AnomalyDetectionPage />} />
            <Route path="/mla-scoring" element={<MLAScoringPage />} />
            <Route path="/forecasting" element={<ForecastingPage />} />
            <Route path="/risk-assessment" element={<RiskAssessmentPage />} />
            <Route path="/monitoring" element={<PerformanceMonitoringPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
