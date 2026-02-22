import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function ROIPredictionPage() {
  const [investment, setInvestment] = useState(1000000)
  const [duration, setDuration] = useState(5)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  const predictROI = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setPrediction({
        roi: 2.4,
        confidence: 0.95,
        lower: 2.1,
        upper: 2.7,
        explanation: ['Market demand: +35%', 'Infrastructure: +20%', 'Competition: -10%']
      })
      setLoading(false)
    }, 1500)
  }

  const chartData = prediction ? [
    { year: 1, value: investment * 1.2 },
    { year: 2, value: investment * 1.5 },
    { year: 3, value: investment * 1.8 },
    { year: 4, value: investment * 2.1 },
    { year: 5, value: investment * prediction.roi },
  ] : []

  return (
    <div className="page-container single">
      <h1>ROI Prediction</h1>

      <div className="content-grid">
        <div className="form-section">
          <h2>Investment Details</h2>
          
          <div className="form-group">
            <label>Investment Amount (₹)</label>
            <input
              type="number"
              value={investment}
              onChange={(e) => setInvestment(Number(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Duration (years)</label>
            <input
              type="number"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
            />
          </div>

          <button className="btn-primary" onClick={predictROI} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict ROI'}
          </button>

          {prediction && (
            <div className="prediction-result">
              <h3>Predicted ROI</h3>
              <div className="roi-value">{prediction.roi}x</div>
              <div className="confidence">95% CI: [{prediction.lower}x - {prediction.upper}x]</div>
              
              <h4>Key Factors</h4>
              <ul>
                {prediction.explanation.map((factor, i) => (
                  <li key={i}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {prediction && (
          <div className="chart-section">
            <h2>Projected Growth</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" label={{ value: 'Year', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Value (₹)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#667eea" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

export default ROIPredictionPage
