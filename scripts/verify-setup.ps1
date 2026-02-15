# PowerShell verification script for project infrastructure setup

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Project Infrastructure Verification" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

function Check-File {
    param($Path)
    if (Test-Path $Path -PathType Leaf) {
        Write-Host "✓ $Path" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ $Path (missing)" -ForegroundColor Red
        return $false
    }
}

function Check-Dir {
    param($Path)
    if (Test-Path $Path -PathType Container) {
        Write-Host "✓ $Path/" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ $Path/ (missing)" -ForegroundColor Red
        return $false
    }
}

Write-Host "Checking project structure..." -ForegroundColor Yellow
Write-Host ""

# Root files
Write-Host "Root Configuration Files:" -ForegroundColor Cyan
Check-File "Cargo.toml"
Check-File "docker-compose.yml"
Check-File ".gitignore"
Check-File ".env.example"
Check-File "README.md"
Check-File "LICENSE"
Check-File "Makefile"
Write-Host ""

# Backend
Write-Host "Backend (Rust):" -ForegroundColor Cyan
Check-Dir "backend"
Check-File "backend/Cargo.toml"
Check-File "backend/Dockerfile"
Check-Dir "backend/src"
Check-File "backend/src/main.rs"
Write-Host ""

# Analytics
Write-Host "Analytics Engine (Julia):" -ForegroundColor Cyan
Check-Dir "analytics"
Check-File "analytics/Project.toml"
Check-File "analytics/Dockerfile"
Check-Dir "analytics/src"
Check-File "analytics/src/AnalyticsEngine.jl"
Write-Host ""

# Database
Write-Host "Database:" -ForegroundColor Cyan
Check-Dir "database"
Check-File "database/init.sql"
Write-Host ""

# Documentation
Write-Host "Documentation:" -ForegroundColor Cyan
Check-Dir "docs"
Check-File "docs/ARCHITECTURE.md"
Check-File "docs/SETUP.md"
Check-File "CONTRIBUTING.md"
Write-Host ""

# Check Docker
Write-Host "Docker Environment:" -ForegroundColor Cyan
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker installed" -ForegroundColor Green
    Write-Host "  $dockerVersion" -ForegroundColor Gray
} catch {
    Write-Host "✗ Docker not found" -ForegroundColor Red
}

try {
    $composeVersion = docker-compose --version
    Write-Host "✓ Docker Compose installed" -ForegroundColor Green
    Write-Host "  $composeVersion" -ForegroundColor Gray
} catch {
    Write-Host "✗ Docker Compose not found" -ForegroundColor Red
}
Write-Host ""

# Check .env
Write-Host "Environment Configuration:" -ForegroundColor Cyan
if (Test-Path ".env" -PathType Leaf) {
    Write-Host "✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "! .env file not found (copy from .env.example)" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Verification Complete" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy .env.example to .env and configure"
Write-Host "2. Run 'docker-compose up -d' to start services"
Write-Host "3. Run 'docker-compose ps' to verify services are running"
