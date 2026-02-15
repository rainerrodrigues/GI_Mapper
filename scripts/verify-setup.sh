#!/bin/bash

# Verification script for project infrastructure setup

echo "==================================="
echo "Project Infrastructure Verification"
echo "==================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${RED}✗${NC} $1/ (missing)"
        return 1
    fi
}

echo "Checking project structure..."
echo ""

# Root files
echo "Root Configuration Files:"
check_file "Cargo.toml"
check_file "docker-compose.yml"
check_file ".gitignore"
check_file ".env.example"
check_file "README.md"
check_file "LICENSE"
check_file "Makefile"
echo ""

# Backend
echo "Backend (Rust):"
check_dir "backend"
check_file "backend/Cargo.toml"
check_file "backend/Dockerfile"
check_dir "backend/src"
check_file "backend/src/main.rs"
echo ""

# Analytics
echo "Analytics Engine (Julia):"
check_dir "analytics"
check_file "analytics/Project.toml"
check_file "analytics/Dockerfile"
check_dir "analytics/src"
check_file "analytics/src/AnalyticsEngine.jl"
echo ""

# Database
echo "Database:"
check_dir "database"
check_file "database/init.sql"
echo ""

# Documentation
echo "Documentation:"
check_dir "docs"
check_file "docs/ARCHITECTURE.md"
check_file "docs/SETUP.md"
check_file "CONTRIBUTING.md"
echo ""

# Check Docker
echo "Docker Environment:"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker installed"
    docker --version
else
    echo -e "${RED}✗${NC} Docker not found"
fi

if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose installed"
    docker-compose --version
else
    echo -e "${RED}✗${NC} Docker Compose not found"
fi
echo ""

# Check .env
echo "Environment Configuration:"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"
else
    echo -e "${YELLOW}!${NC} .env file not found (copy from .env.example)"
fi
echo ""

echo "==================================="
echo "Verification Complete"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure"
echo "2. Run 'docker-compose up -d' to start services"
echo "3. Run 'docker-compose ps' to verify services are running"
