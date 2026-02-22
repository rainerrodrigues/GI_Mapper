import React, { useState } from 'react'
import { MapContainer, TileLayer, Polygon, CircleMarker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

function ClusteringPage() {
  const [algorithm, setAlgorithm] = useState('auto')
  const [loading, setLoading] = useState(false)
  const [clusters, setClusters] = useState([])

  const detectClusters = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setClusters([
        { id: 1, center: [28.6139, 77.2090], members: 15, value: 12500 },
        { id: 2, center: [19.0760, 72.8777], members: 12, value: 9800 },
      ])
      setLoading(false)
    }, 1500)
  }

  return (
    <div className="page-container">
      <div className="sidebar">
        <h2>Cluster Detection</h2>
        
        <div className="form-group">
          <label>Algorithm</label>
          <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
            <option value="auto">Auto (Best)</option>
            <option value="dbscan">DBSCAN</option>
            <option value="kmeans">K-Means</option>
          </select>
        </div>

        <button className="btn-primary" onClick={detectClusters} disabled={loading}>
          {loading ? 'Detecting...' : 'Detect Clusters'}
        </button>

        {clusters.length > 0 && (
          <div className="results">
            <h3>Results ({clusters.length} clusters)</h3>
            {clusters.map((cluster) => (
              <div key={cluster.id} className="result-card">
                <h4>Cluster {cluster.id}</h4>
                <p>Members: {cluster.members}</p>
                <p>Value: ₹{cluster.value}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="map-container">
        <MapContainer center={[20.5937, 78.9629]} zoom={5} style={{ height: '100%', width: '100%' }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {clusters.map((cluster) => (
            <CircleMarker key={cluster.id} center={cluster.center} radius={10}>
              <Popup>Cluster {cluster.id}</Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  )
}

export default ClusteringPage
