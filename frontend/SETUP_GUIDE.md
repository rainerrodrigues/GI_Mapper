# Frontend Dashboard Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The dashboard will open at http://localhost:5173

### 3. View the Dashboard

Navigate through the sidebar to explore all 7 AI features:
- Dashboard (overview)
- Clustering (spatial analysis)
- ROI Prediction (investment forecasting)
- Anomaly Detection (fraud detection)
- MLA Scoring (development assessment)
- Forecasting (time series)
- Risk Assessment (project risk)
- Monitoring (model performance)

## Configuration

### Backend API URL

By default, the dashboard connects to http://localhost:3000/api/v1

To change this, create a `.env` file:

```
VITE_API_URL=http://your-backend-url/api/v1
```

## Features

### Current Implementation
- ✅ All 8 pages created
- ✅ Navigation with sidebar
- ✅ Responsive layouts
- ✅ Chart visualizations (Recharts)
- ✅ Map integration (Leaflet)
- ✅ API service layer
- ✅ Simulated data for demo

### Ready for Integration
- Backend API endpoints
- Authentication flow
- Real-time data updates
- SHAP explanation displays

## Development Notes

### Simulated Data
All pages currently use simulated data with setTimeout to mimic API calls. This allows you to:
- Test the UI without backend
- See loading states
- Understand data structures

### API Integration
When ready to connect to real backend:
1. Ensure backend is running on port 3000
2. Update API calls in page components
3. Replace setTimeout with actual API calls from services/api.js

Example:
```javascript
// Current (simulated)
setTimeout(() => {
  setData(mockData)
}, 1500)

// Real API
import { roiAPI } from '../services/api'
const response = await roiAPI.predict(formData)
setData(response.data)
```

## Troubleshooting

### Port Already in Use
If port 5173 is busy:
```bash
npm run dev -- --port 5174
```

### Module Not Found
Clear node_modules and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Leaflet CSS Not Loading
The Leaflet CSS is loaded in index_react.html. If maps don't display correctly, check browser console for CSS errors.

## Next Steps

1. Test all pages in the dashboard
2. Connect to backend API (when ready)
3. Add authentication
4. Customize styling as needed
5. Add more interactive features

## Support

For issues or questions, refer to:
- Main README.md
- OPTION_A_COMPLETE.md (AI features documentation)
- Backend API documentation
