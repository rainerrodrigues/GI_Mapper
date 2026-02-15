# AI-Powered GIS Platform - Demo Startup Script
# This script starts all required services for the demo

Write-Host "🌍 AI-Powered Blockchain GIS Platform - Demo Startup" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Start PostgreSQL database
Write-Host ""
Write-Host "Starting PostgreSQL database..." -ForegroundColor Yellow
docker-compose up -d postgres

Write-Host "Waiting for database to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check database health
Write-Host "Checking database connection..." -ForegroundColor Yellow
$dbReady = $false
for ($i = 1; $i -le 5; $i++) {
    try {
        docker exec gi_mapper-postgres-1 pg_isready -U postgres | Out-Null
        $dbReady = $true
        break
    } catch {
        Write-Host "  Attempt $i/5 - Database not ready yet..." -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if ($dbReady) {
    Write-Host "✓ Database is ready" -ForegroundColor Green
} else {
    Write-Host "✗ Database failed to start" -ForegroundColor Red
    exit 1
}

# Start backend server
Write-Host ""
Write-Host "Starting Rust backend server..." -ForegroundColor Yellow
Write-Host "  API will be available at http://localhost:3000" -ForegroundColor Gray
Write-Host ""

# Change to backend directory and run
Set-Location backend

Write-Host "Building and starting backend (this may take a moment)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Backend server is starting..." -ForegroundColor Cyan
Write-Host "Once you see 'Server listening on 0.0.0.0:3000'," -ForegroundColor Cyan
Write-Host "open frontend/index.html in your browser" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

cargo run

# Note: The script will stay running while the backend is active
# Press Ctrl+C to stop the backend server
