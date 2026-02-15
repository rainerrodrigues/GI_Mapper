# Test API Script - Verify the backend is working

Write-Host "🧪 Testing AI-Powered GIS Platform API" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:3000"

# Test 1: Health check (if available)
Write-Host "Test 1: Checking if server is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$baseUrl/api/v1/gi-products" -Method GET -ErrorAction Stop
    Write-Host "✓ Server is responding" -ForegroundColor Green
} catch {
    Write-Host "✗ Server is not responding. Make sure the backend is running." -ForegroundColor Red
    Write-Host "  Run: .\start-demo.ps1" -ForegroundColor Gray
    exit 1
}

# Test 2: Cluster detection with sample data
Write-Host ""
Write-Host "Test 2: Testing cluster detection..." -ForegroundColor Yellow

$sampleData = @{
    data_points = @(
        @{ id = "p1"; latitude = 28.6139; longitude = 77.2090; value = 100.0 },
        @{ id = "p2"; latitude = 28.7041; longitude = 77.1025; value = 150.0 },
        @{ id = "p3"; latitude = 28.5355; longitude = 77.3910; value = 120.0 },
        @{ id = "p4"; latitude = 19.0760; longitude = 72.8777; value = 200.0 },
        @{ id = "p5"; latitude = 19.1136; longitude = 72.8697; value = 180.0 },
        @{ id = "p6"; latitude = 12.9716; longitude = 77.5946; value = 160.0 },
        @{ id = "p7"; latitude = 12.9141; longitude = 77.6411; value = 140.0 },
        @{ id = "p8"; latitude = 13.0827; longitude = 80.2707; value = 130.0 }
    )
    algorithm = "auto"
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/clusters/detect" `
        -Method POST `
        -ContentType "application/json" `
        -Body $sampleData `
        -ErrorAction Stop
    
    Write-Host "✓ Cluster detection successful" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  Algorithm used: $($response.algorithm_used)" -ForegroundColor White
    Write-Host "  Clusters found: $($response.clusters.Count)" -ForegroundColor White
    Write-Host "  Silhouette score: $($response.quality_metrics.silhouette_score)" -ForegroundColor White
    Write-Host "  Model version: $($response.model_version)" -ForegroundColor White
    
    Write-Host ""
    Write-Host "Cluster details:" -ForegroundColor Cyan
    foreach ($cluster in $response.clusters) {
        Write-Host "  Cluster $($cluster.cluster_id):" -ForegroundColor Yellow
        Write-Host "    Members: $($cluster.member_count)" -ForegroundColor White
        Write-Host "    Density: $([math]::Round($cluster.density, 4))" -ForegroundColor White
        Write-Host "    Economic value: ₹$([math]::Round($cluster.economic_value, 2))" -ForegroundColor White
        Write-Host "    Centroid: [$([math]::Round($cluster.centroid[0], 4)), $([math]::Round($cluster.centroid[1], 4))]" -ForegroundColor White
    }
    
} catch {
    Write-Host "✗ Cluster detection failed" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test 3: List clusters
Write-Host ""
Write-Host "Test 3: Listing stored clusters..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/clusters" -Method GET -ErrorAction Stop
    Write-Host "✓ Successfully retrieved clusters" -ForegroundColor Green
    Write-Host "  Total clusters in database: $($response.total)" -ForegroundColor White
} catch {
    Write-Host "✗ Failed to list clusters" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✓ All tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open frontend/index.html in your browser" -ForegroundColor White
Write-Host "  2. Click 'Detect Clusters' to see the visualization" -ForegroundColor White
Write-Host "  3. Try different algorithms (DBSCAN, K-Means, Auto)" -ForegroundColor White
Write-Host ""
