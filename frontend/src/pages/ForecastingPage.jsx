import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart } from 'recharts'

function ForecastingPage() {
  const [metric, setMetric] = useState('revenue')
  const [horizon, setHorizon] = useState(5)
  const [loading, setLoading] = useState(false)
  const [forecast, setForecast] = useState(null)

  const generateForecast = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      const historical = [
        { year: 2020, actual: 100, forecast: null, lower: null, upper: null },
        { year: 2021, actual: 120, forecast: null, lower: null, upper: null },
        { year: 2022, actual: 145, forecast: null, lower: null, upper: null },
        { year: 2023, actual: 160, forecast: null, lower: null, upper: null },
        { year: 2024, actual: 180, forecast: null, lower: null, upper: null },
      ]
      const future = [
        { year: 2025, actual: null, forecast: 205, lower: 195, upper: 215 },
        { year: 2026, actual: null, forecast: 235, lower: 220, upper: 250 },
        { year: 2027, actual: null, forecast: 270, lower: 250, upper: 290 },
        { year: 2028, actual: null, forecast: 310, lower: 285, upper: 335 },
        { year: 2029, actual: null, forecast: 355, lower: 325, upper: 385 },
      ]
      setForecast({
        data: [...historical, ...future],
        models: ['ARIMA', 'Prophet', 'LSTM'],
        confidence: 0.95
      })
      setLoading(false)
    }, 1500)
  }

  return (
    <div className="page-container single">
      <h1>Time Series Forecasting</h1>

      <div className="content-grid">
        <div className="form-section">
          <h2>Forecast Settings</h2>
          
          <div className="form-group">
            <label>Metric</label>
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option value="revenue">Revenue</option>
              <option value="production">Production</option>
              <option value="employment">Employment</option>
            </select>
          </div>

          <div className="form-group">
            <label>Horizon (years)</label>
            <input
              type="number"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              min="1"
              max="10"
            />
          </div>

          <div className="form-group">
            <label>Confidence Level</label>
            <select>
              <option>95%</option>
              <option>80%</option>
            </select>
          </div>

          <button className="btn-primary" onClick={generateForecast} disabled={loading}>
            {loading ? 'Forecasting...' : 'Generate Forecast'}
          </button>

          {forecast && (
            <div className="forecast-info">
              <h3>Forecast Details</h3>
              <p><strong>Models Used:</strong> {forecast.models.join(', ')}</p>
              <p><strong>Confidence:</strong> {(forecast.confidence * 100).toFixed(0)}%</p>
              <p><strong>Trend:</strong> Upward</p>
              <p><strong>Seasonality:</strong> Detected</p>
            </div>
          )}
        </div>

        {forecast && (
          <div className="chart-section">
            <h2>Forecast Visualization</h2>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={forecast.data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="upper"
                  fill="#667eea"
                  stroke="none"
                  fillOpacity={0.2}
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  fill="#fff"
                  stroke="none"
                  fillOpacity={1}
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#2ecc71"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Historical"
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#667eea"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={{ r: 4 }}
                  name="Forecast"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

export default ForecastingPage
