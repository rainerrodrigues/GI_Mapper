import React, { useState } from 'react'
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, ResponsiveContainer } from 'recharts'

function RiskAssessmentPage() {
  const [projectType, setProjectType] = useState('')
  const [loading, setLoading] = useState(false)
  const [assessment, setAssessment] = useState(null)

  const assessRisk = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setAssessment({
        overall: 'Medium',
        probability: 0.42,
        confidence: 0.88,
        dimensions: [
          { dimension: 'Financial', score: 45, fullMark: 100 },
          { dimension: 'Operational', score: 38, fullMark: 100 },
          { dimension: 'Social', score: 52, fullMark: 100 },
          { dimension: 'Environmental', score: 30, fullMark: 100 },
          { dimension: 'Regulatory', score: 48, fullMark: 100 },
        ],
        mitigations: [
          'Diversify funding sources to reduce financial risk',
          'Implement robust project management framework',
          'Engage local communities early in planning',
          'Conduct environmental impact assessment'
        ]
      })
      setLoading(false)
    }, 1500)
  }

  const getRiskColor = (level) => {
    const colors = {
      'Low': '#2ecc71',
      'Medium': '#f39c12',
      'High': '#e74c3c',
      'Critical': '#c0392b'
    }
    return colors[level] || '#95a5a6'
  }

  return (
    <div className="page-container single">
      <h1>Risk Assessment</h1>

      <div className="content-grid">
        <div className="form-section">
          <h2>Project Details</h2>
          
          <div className="form-group">
            <label>Project Type</label>
            <select value={projectType} onChange={(e) => setProjectType(e.target.value)}>
              <option value="">Select type</option>
              <option value="infrastructure">Infrastructure</option>
              <option value="agriculture">Agriculture</option>
              <option value="manufacturing">Manufacturing</option>
              <option value="services">Services</option>
            </select>
          </div>

          <div className="form-group">
            <label>Location</label>
            <input type="text" placeholder="Enter location" />
          </div>

          <div className="form-group">
            <label>Investment (₹)</label>
            <input type="number" placeholder="Enter amount" />
          </div>

          <button className="btn-primary" onClick={assessRisk} disabled={loading}>
            {loading ? 'Assessing...' : 'Assess Risk'}
          </button>

          {assessment && (
            <div className="risk-result">
              <h3>Risk Level</h3>
              <div className="risk-badge" style={{ backgroundColor: getRiskColor(assessment.overall) }}>
                {assessment.overall}
              </div>
              <div className="risk-probability">
                Probability: {(assessment.probability * 100).toFixed(0)}%
              </div>
              <div className="risk-confidence">
                Confidence: {(assessment.confidence * 100).toFixed(0)}%
              </div>
            </div>
          )}
        </div>

        {assessment && (
          <div className="chart-section">
            <h2>Risk Dimensions</h2>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={assessment.dimensions}>
                <PolarGrid />
                <PolarAngleAxis dataKey="dimension" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="Risk Score" dataKey="score" stroke="#667eea" fill="#667eea" fillOpacity={0.6} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>

            <div className="mitigations">
              <h3>Recommended Mitigations</h3>
              <ul>
                {assessment.mitigations.map((mitigation, i) => (
                  <li key={i}>{mitigation}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default RiskAssessmentPage
