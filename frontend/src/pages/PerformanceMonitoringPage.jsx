import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function PerformanceMonitoringPage() {
  const [selectedModel, setSelectedModel] = useState('roi_predictor')
  
  const models = [
    { id: 'roi_predictor', name: 'ROI Predictor', accuracy: 0.94, status: 'healthy' },
    { id: 'anomaly_detector', name: 'Anomaly Detector', accuracy: 0.91, status: 'healthy' },
    { id: 'risk_assessor', name: 'Risk Assessor', accuracy: 0.88, status: 'warning' },
    { id: 'forecaster', name: 'Forecaster', accuracy: 0.92, status: 'healthy' },
  ]

  const performanceData = [
    { date: '2024-01', accuracy: 0.92, latency: 120 },
    { date: '2024-02', accuracy: 0.93, latency: 115 },
    { date: '2024-03', accuracy: 0.94, latency: 110 },
  ]

  return (
    <div className="page-container single">
      <h1>Model Performance Monitoring</h1>

      <div className="models-grid">
        {models.map((model) => (
          <div
            key={model.id}
            className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
            onClick={() => setSelectedModel(model.id)}
          >
            <h3>{model.name}</h3>
            <div className="model-accuracy">{(model.accuracy * 100).toFixed(1)}%</div>
            <div className={`model-status ${model.status}`}>{model.status}</div>
          </div>
        ))}
      </div>

      <div className="metrics-section">
        <h2>Performance Metrics</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#667eea" strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#f39c12" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <h4>Predictions/Day</h4>
          <div className="kpi-value">1,247</div>
        </div>
        <div className="kpi-card">
          <h4>Avg Latency</h4>
          <div className="kpi-value">110ms</div>
        </div>
        <div className="kpi-card">
          <h4>Drift Detected</h4>
          <div className="kpi-value">No</div>
        </div>
        <div className="kpi-card">
          <h4>Last Retrain</h4>
          <div className="kpi-value">2 days ago</div>
        </div>
      </div>
    </div>
  )
}

export default PerformanceMonitoringPage
