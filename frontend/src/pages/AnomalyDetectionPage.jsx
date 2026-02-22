import React, { useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

function AnomalyDetectionPage() {
  const [loading, setLoading] = useState(false)
  const [anomalies, setAnomalies] = useState([])

  const detectAnomalies = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setAnomalies([
        { id: 1, score: 0.92, severity: 'critical', type: 'Transaction', value: 150000, x: 10, y: 90 },
        { id: 2, score: 0.78, severity: 'high', type: 'Transaction', value: 85000, x: 25, y: 75 },
        { id: 3, score: 0.65, severity: 'medium', type: 'Transaction', value: 45000, x: 40, y: 60 },
      ])
      setLoading(false)
    }, 1500)
  }

  const getSeverityColor = (severity) => {
    const colors = {
      critical: '#e74c3c',
      high: '#f39c12',
      medium: '#f1c40f',
      low: '#3498db'
    }
    return colors[severity] || '#95a5a6'
  }

  return (
    <div className="page-container single">
      <h1>Anomaly Detection</h1>

      <div className="content-grid">
        <div className="form-section">
          <h2>Detection Settings</h2>
          
          <div className="form-group">
            <label>Data Source</label>
            <select>
              <option>All Transactions</option>
              <option>GI Products</option>
              <option>Clusters</option>
            </select>
          </div>

          <div className="form-group">
            <label>Sensitivity</label>
            <select>
              <option>High</option>
              <option>Medium</option>
              <option>Low</option>
            </select>
          </div>

          <button className="btn-primary" onClick={detectAnomalies} disabled={loading}>
            {loading ? 'Detecting...' : 'Detect Anomalies'}
          </button>

          {anomalies.length > 0 && (
            <div className="results">
              <h3>Detected Anomalies ({anomalies.length})</h3>
              {anomalies.map((anomaly) => (
                <div key={anomaly.id} className="anomaly-card" style={{ borderLeftColor: getSeverityColor(anomaly.severity) }}>
                  <div className="anomaly-header">
                    <span className="anomaly-id">#{anomaly.id}</span>
                    <span className={`severity-badge ${anomaly.severity}`}>{anomaly.severity}</span>
                  </div>
                  <p>Score: {anomaly.score.toFixed(2)}</p>
                  <p>Type: {anomaly.type}</p>
                  <p>Value: ₹{anomaly.value.toLocaleString()}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        {anomalies.length > 0 && (
          <div className="chart-section">
            <h2>Anomaly Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" name="Feature 1" />
                <YAxis dataKey="y" name="Feature 2" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Anomalies" data={anomalies}>
                  {anomalies.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getSeverityColor(entry.severity)} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

export default AnomalyDetectionPage
