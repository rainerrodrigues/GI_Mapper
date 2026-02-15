# Frontend Dashboard - AI-Powered GIS Platform

## Overview

This is a lightweight, single-page application (SPA) that provides an interactive map interface for visualizing economic clusters detected by the AI-powered GIS platform.

## Technology Stack

- **HTML5** - Structure
- **CSS3** - Styling with modern gradients and responsive design
- **Vanilla JavaScript** - No framework dependencies
- **Leaflet.js** - Interactive map library
- **OpenStreetMap** - Map tiles

## Features

### 1. Interactive Map
- Pan and zoom across India
- Click clusters for details
- Popup information windows
- Responsive to window size

### 2. Cluster Visualization
- Color-coded cluster boundaries
- Cluster centroids marked
- Member points displayed
- Auto-fit to show all clusters

### 3. Algorithm Selection
- DBSCAN (density-based)
- K-Means (centroid-based)
- Auto (runs both, picks best)

### 4. Results Display
- Quality metrics dashboard
- Cluster statistics
- Member counts
- Economic values
- Density calculations

## How to Use

### Option 1: Direct Open
Simply double-click `index.html` or open it in your browser:
```powershell
start index.html
```

### Option 2: HTTP Server
For better CORS handling, serve with an HTTP server:

**Python:**
```powershell
python -m http.server 8080
```

**Node.js (if installed):**
```powershell
npx http-server -p 8080
```

Then open: http://localhost:8080

## Configuration

### API Endpoint
The frontend connects to the backend API at:
```javascript
http://localhost:3000/api/v1/clusters/detect
```

To change this, edit line ~200 in `index.html`:
```javascript
const response = await fetch('http://YOUR_API_URL/api/v1/clusters/detect', {
```

### Map Center
Default center is India (20.5937°N, 78.9629°E). To change:
```javascript
const map = L.map('map').setView([YOUR_LAT, YOUR_LON], ZOOM_LEVEL);
```

### Sample Data
The demo generates sample data for 5 Indian cities. To modify:
```javascript
const regions = [
    { name: 'City Name', lat: XX.XXXX, lon: YY.YYYY, count: N },
    // Add more regions...
];
```

## File Structure

```
frontend/
├── index.html          # Main application file
└── README.md          # This file
```

Everything is contained in a single HTML file for simplicity.

## API Integration

### Request Format
```json
{
  "data_points": [
    {
      "id": "point1",
      "latitude": 28.6139,
      "longitude": 77.2090,
      "value": 100.0
    }
  ],
  "algorithm": "auto"
}
```

### Response Format
```json
{
  "clusters": [
    {
      "id": "uuid",
      "cluster_id": 1,
      "boundary": [[lon, lat], ...],
      "centroid": [lon, lat],
      "member_count": 10,
      "density": 0.5,
      "economic_value": 1000.0,
      "member_ids": ["point1", ...]
    }
  ],
  "algorithm_used": "K-Means",
  "quality_metrics": {
    "silhouette_score": 0.65,
    "davies_bouldin_index": 0.85,
    "calinski_harabasz_score": 150.0
  },
  "model_version": "0.2.0"
}
```

## Customization

### Colors
Cluster colors are defined in the `colors` array:
```javascript
const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
```

### Styling
All CSS is embedded in the `<style>` section. Key classes:
- `.header` - Top banner
- `.sidebar` - Left control panel
- `.map-container` - Map area
- `.cluster-item` - Result items
- `.metrics` - Quality metrics box

### Map Tiles
To use different map tiles (e.g., satellite view):
```javascript
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri'
}).addTo(map);
```

## Browser Compatibility

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

Requires:
- ES6 JavaScript support
- Fetch API
- CSS Grid
- CSS Flexbox

## Performance

- Initial load: <1 second
- Map rendering: <100ms
- API call: ~100-500ms
- Cluster drawing: <100ms
- Total interaction: <1 second

## Troubleshooting

### Map not loading
- Check internet connection (needs to download map tiles)
- Check browser console for errors
- Verify Leaflet CDN is accessible

### API errors
- Ensure backend is running on port 3000
- Check CORS configuration in backend
- Verify API endpoint URL
- Check browser console for network errors

### Clusters not displaying
- Verify API response format
- Check that coordinates are valid
- Ensure boundary arrays are not empty
- Look for JavaScript errors in console

### CORS errors
```
Access to fetch at 'http://localhost:3000' from origin 'file://' has been blocked by CORS policy
```

**Solution**: Serve the file with an HTTP server instead of opening directly.

## Development

### Adding New Features

1. **Add a new form field:**
```html
<div class="form-group">
    <label for="myField">My Field</label>
    <input type="text" id="myField" />
</div>
```

2. **Read the value:**
```javascript
const myValue = document.getElementById('myField').value;
```

3. **Include in API request:**
```javascript
body: JSON.stringify({
    data_points: dataPoints,
    algorithm: algorithm,
    my_field: myValue
})
```

### Adding New Visualizations

1. **Create a new layer:**
```javascript
const myLayer = L.layerGroup().addTo(map);
```

2. **Add features:**
```javascript
L.marker([lat, lon]).addTo(myLayer);
L.polygon(coords).addTo(myLayer);
```

3. **Add to cleanup:**
```javascript
myLayers.push(myLayer);
```

## Future Enhancements

Planned improvements:
- [ ] Add layer toggle controls
- [ ] Implement data upload (CSV, GeoJSON)
- [ ] Add export functionality
- [ ] Show SHAP explanations
- [ ] Add time series visualization
- [ ] Implement user authentication
- [ ] Add real-time updates
- [ ] Mobile responsive improvements

## Production Considerations

For production deployment:

1. **Minify code**: Use a build tool to minify HTML/CSS/JS
2. **CDN**: Host Leaflet locally instead of using CDN
3. **Environment variables**: Use config file for API URL
4. **Error tracking**: Add Sentry or similar
5. **Analytics**: Add Google Analytics or similar
6. **HTTPS**: Serve over HTTPS
7. **Caching**: Implement service worker for offline support

## License

See main project LICENSE file.

## Support

For issues or questions:
- Check browser console for errors
- Review DEMO_GUIDE.md
- Check backend logs
- Refer to main project README.md

---

**Built with ❤️ for Rural India Economic Development**
