import React from 'react'

function Dashboard() {
  const features = [
    { name: 'Clustering', icon: '🗺️', status: 'Active', count: '12 clusters' },
    { name: 'ROI Prediction', icon: '💰', status: 'Active', count: '45 predictions' },
    { name: 'Anomaly Detection', icon: '🔍', status: 'Active', count: '3 anomalies' },
    { name: 'MLA Scoring', icon: '📈', status: 'Active', count: '543 scores' },
    { name: 'Forecasting', icon: '📉', status: 'Active', count: '8 forecasts' },
    { name: 'Risk Assessment', icon: '⚠️', status: 'Active', count: '23 assessments' },
    { name: 'Monitoring', icon: '📡', status: 'Active', count: '7 models' },
  ]

  return (
    <div className="dashboard">
      <h1>Dashboard</h1>
      <p className="subtitle">AI-Powered GIS Platform for Rural India</p>

      <div className="feature-grid">
        {features.map((feature) => (
          <div key={feature.name} className="feature-card">
            <div className="feature-icon">{feature.icon}</div>
            <h3>{feature.name}</h3>
            <div className="feature-status">{feature.status}</div>
            <div className="feature-count">{feature.count}</div>
          </div>
        ))}
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <h4>Total GI Products</h4>
          <div className="stat-value">127</div>
        </div>
        <div className="stat-card">
          <h4>Economic Clusters</h4>
          <div className="stat-value">12</div>
        </div>
        <div className="stat-card">
          <h4>Predictions Made</h4>
          <div className="stat-value">45</div>
        </div>
        <div className="stat-card">
          <h4>Model Accuracy</h4>
          <div className="stat-value">94.2%</div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
