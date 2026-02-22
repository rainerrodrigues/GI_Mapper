# AI-Powered GIS Platform - React Dashboard

## Overview

Comprehensive React-based dashboard for visualizing all 7 AI/ML features of the platform.

## Features Implemented

### 1. Dashboard (Home)
- Overview of all 7 AI features
- Summary statistics
- Feature status indicators

### 2. Clustering Page
- Interactive map with Leaflet
- DBSCAN and K-Means algorithm selection
- Cluster visualization with boundaries
- Cluster statistics display

### 3. ROI Prediction Page
- Investment input form
- ROI prediction with confidence intervals
- Growth projection chart
- SHAP explanation display

### 4. Anomaly Detection Page
- Anomaly detection settings
- Severity classification (critical, high, medium, low)
- Scatter plot visualization
- Anomaly details with scores

### 5. MLA Scoring Page
- Constituency selection
- Development score computation
- Score breakdown by dimension
- Indicator weights table

### 6. Forecasting Page
- Time series forecasting
- Multi-horizon predictions (1-10 years)
- Confidence intervals (80%, 95%)
- ARIMA, Prophet, LSTM ensemble

### 7. Risk Assessment Page
- Multi-dimensional risk scoring
- Radar chart visualization
- Risk mitigation recommendations
- Calibrated probability estimation

### 8. Performance Monitoring Page
- Model performance metrics
- Accuracy and latency tracking
- KPI dashboard
- Model health status

## Tech Stack

- React 18.2
- React Router 6.20
- Leaflet + React-Leaflet (maps)
- Recharts (charts)
- Axios (API calls)
- Vite (build tool)

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

The dashboard will be available at http://localhost:5173

## Build

```bash
npm run build
```

## Project Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── Dashboard.jsx
│   │   ├── ClusteringPage.jsx
│   │   ├── ROIPredictionPage.jsx
│   │   ├── AnomalyDetectionPage.jsx
│   │   ├── MLAScoringPage.jsx
│   │   ├── ForecastingPage.jsx
│   │   ├── RiskAssessmentPage.jsx
│   │   └── PerformanceMonitoringPage.jsx
│   ├── services/
│   │   └── api.js
│   ├── App.jsx
│   ├── App.css
│   ├── main.jsx
│   └── index.css
├── index_react.html
├── package.json
└── vite.config.js
```

## API Integration

The dashboard is configured to connect to the backend API at:
- Development: http://localhost:3000/api/v1
- Production: Set VITE_API_URL environment variable

All API calls are handled through the `services/api.js` module with:
- Automatic JWT token injection
- Error handling and 401 redirect
- Centralized API endpoints

## Features

### Navigation
- Collapsible sidebar with icons
- Active route highlighting
- Smooth transitions

### Responsive Design
- Grid-based layouts
- Flexible components
- Mobile-friendly (future enhancement)

### Data Visualization
- Interactive maps with Leaflet
- Charts with Recharts
- Real-time updates (simulated)

### User Experience
- Loading states
- Error handling
- Intuitive forms
- Clear feedback

## Next Steps

1. Connect to real backend API endpoints
2. Add authentication flow
3. Implement WebSocket for real-time updates
4. Add data export functionality
5. Enhance mobile responsiveness
6. Add unit tests with Vitest
7. Add E2E tests with Playwright

## Notes

- Currently uses simulated data for demo purposes
- Backend API integration ready via services/api.js
- All 7 AI features have dedicated pages
- Follows established design patterns from existing demo
