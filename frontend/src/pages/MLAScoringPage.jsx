import React, { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function MLAScoringPage() {
  const [constituency, setConstituency] = useState('')
  const [loading, setLoading] = useState(false)
  const [score, setScore] = useState(null)

  const computeScore = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setScore({
        total: 78.5,
        confidence: [72.1, 84.9],
        dimensions: [
          { name: 'Infrastructure', score: 82, weight: 0.3 },
          { name: 'Education', score: 75, weight: 0.25 },
          { name: 'Healthcare', score: 80, weight: 0.2 },
          { name: 'Employment', score: 76, weight: 0.15 },
          { name: 'Agriculture', score: 78, weight: 0.1 },
        ],
        insights: [
          'Infrastructure development above state average',
          'Education indicators show steady improvement',
          'Healthcare access needs attention'
        ]
      })
      setLoading(false)
    }, 1500)
  }

  return (
    <div className="page-container single">
      <h1>MLA Development Scoring</h1>

      <div className="content-grid">
        <div className="form-section">
          <h2>Constituency Selection</h2>
          
          <div className="form-group">
            <label>Constituency</label>
            <input
              type="text"
              placeholder="Enter constituency name"
              value={constituency}
              onChange={(e) => setConstituency(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>State</label>
            <select>
              <option>Select state</option>
              <option>Uttar Pradesh</option>
              <option>Maharashtra</option>
              <option>Bihar</option>
              <option>West Bengal</option>
            </select>
          </div>

          <button className="btn-primary" onClick={computeScore} disabled={loading}>
            {loading ? 'Computing...' : 'Compute Score'}
          </button>

          {score && (
            <div className="score-result">
              <h3>Development Score</h3>
              <div className="score-value">{score.total.toFixed(1)}</div>
              <div className="score-range">Range: {score.confidence[0].toFixed(1)} - {score.confidence[1].toFixed(1)}</div>
              
              <h4>Key Insights</h4>
              <ul>
                {score.insights.map((insight, i) => (
                  <li key={i}>{insight}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {score && (
          <div className="chart-section">
            <h2>Score Breakdown</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={score.dimensions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="score" fill="#667eea" />
              </BarChart>
            </ResponsiveContainer>

            <div className="weights-table">
              <h3>Indicator Weights</h3>
              <table>
                <thead>
                  <tr>
                    <th>Dimension</th>
                    <th>Score</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {score.dimensions.map((dim, i) => (
                    <tr key={i}>
                      <td>{dim.name}</td>
                      <td>{dim.score}</td>
                      <td>{(dim.weight * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default MLAScoringPage
