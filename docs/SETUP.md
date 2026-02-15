# Development Environment Setup

## Prerequisites

### Required
- Docker 24.0+
- Docker Compose 2.20+
- Git

### Optional (for local development)
- Rust 1.75+
- Julia 1.10+
- PostgreSQL 16+ with PostGIS 3.4+

## Initial Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-blockchain-gis-platform
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Services
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or using Make
make up
```

### 4. Verify Services
```bash
# Check all services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

## Service URLs

- **Backend API**: http://localhost:8000
- **Analytics gRPC**: http://localhost:50051
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **IPFS API**: http://localhost:5001
- **IPFS Gateway**: http://localhost:8080
- **Substrate WebSocket**: ws://localhost:9944
- **Substrate RPC**: http://localhost:9933

## Database Access

```bash
# Using Docker
docker-compose exec postgres psql -U gis_user -d gis_platform

# Or using Make
make db-shell
```

## Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker ps

# Check logs for errors
docker-compose logs

# Restart services
docker-compose restart
```

### Port conflicts
Edit `docker-compose.yml` to change port mappings if needed.

### Database connection issues
Ensure PostgreSQL is healthy:
```bash
docker-compose ps postgres
```

## Development Workflow

### Backend Development
```bash
# Build
cd backend
cargo build

# Run tests
cargo test

# Format code
cargo fmt

# Lint
cargo clippy
```

### Analytics Development
```bash
# Install dependencies
cd analytics
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Or using Make
make down
make clean  # removes volumes
```
