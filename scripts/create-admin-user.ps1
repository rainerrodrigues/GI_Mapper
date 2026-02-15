# PowerShell script to create initial admin user in the database
# Usage: .\scripts\create-admin-user.ps1

Write-Host "Creating initial admin user..." -ForegroundColor Green
Write-Host ""

# Check if Docker is running
$dockerRunning = docker ps 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker is not running. Please start Docker and try again." -ForegroundColor Red
    exit 1
}

# Check if postgres container is running
$postgresRunning = docker ps --filter "name=postgres" --format "{{.Names}}"
if (-not $postgresRunning) {
    Write-Host "Error: PostgreSQL container is not running." -ForegroundColor Red
    Write-Host "Start the services with: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host "PostgreSQL container found: $postgresRunning" -ForegroundColor Cyan

# Execute the SQL script
Write-Host "Executing SQL script to create admin user..." -ForegroundColor Cyan
docker exec -i $postgresRunning psql -U gis_user -d gis_platform -f /scripts/create-admin-user.sql

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Admin user creation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Default credentials:" -ForegroundColor Yellow
    Write-Host "  Username: admin" -ForegroundColor White
    Write-Host "  Password: admin123" -ForegroundColor White
    Write-Host ""
    Write-Host "Additional test users:" -ForegroundColor Yellow
    Write-Host "  Analyst - username: analyst, password: admin123" -ForegroundColor White
    Write-Host "  Viewer  - username: viewer, password: admin123" -ForegroundColor White
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Change these passwords immediately in production!" -ForegroundColor Red
    Write-Host ""
    Write-Host "You can now login at: http://localhost:8080/api/v1/auth/login" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "✗ Failed to create admin user" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
    exit 1
}
