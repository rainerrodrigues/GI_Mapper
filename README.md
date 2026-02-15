# AI-Powered Blockchain GIS Platform for Rural India

An AI-powered blockchain GIS platform that combines spatial analytics, machine learning, blockchain verification, and interactive visualization to support economic development planning in rural India.

## Features

- **GI Product Mapping**: Map Geographical Indication products across regions
- **Economic Cluster Detection**: AI-powered spatial clustering using DBSCAN and K-Means
- **Investment ROI Prediction**: Ensemble regression models for investment decision support
- **MLA Development Impact Scoring**: Weighted AI indices for evaluating development initiatives
- **Anomaly Detection**: Unsupervised learning for identifying spending irregularities
- **Explainable AI**: SHAP-based explanations for all AI outputs
- **Blockchain Verification**: Immutable audit trails using Substrate
- **Interactive Dashboards**: Web-based visualization with interactive maps

## Architecture

- **Backend API**: Rust with Axum framework
- **Analytics Engine**: Julia for high-performance ML/AI computations
- **Database**: PostgreSQL with PostGIS extension
- **Blockchain**: Substrate framework
- **Storage**: IPFS for distributed evidence storage
- **Frontend**: React with TypeScript (coming soon)

## Prerequisites

- Docker and Docker Compose
- Rust 1.75+ (for local development)
- Julia 1.10+ (for local development)
- PostgreSQL 16+ with PostGIS 3.4+ (or use Docker)

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-blockchain-gis-platform
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Start all services:
```bash
docker-compose up -d
```

4. Check service health:
```bash
docker-compose ps
```

5. View logs:
```bash
docker-compose logs -f
```

### Local Development

#### Backend (Rust)

```bash
cd backend
cargo build
cargo run
```

#### Analytics Engine (Julia)

```bash
cd analytics
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using AnalyticsEngine; AnalyticsEngine.start_service()'
```

#### Database Setup

```bash
# Start PostgreSQL with PostGIS
docker-compose up -d postgres

# Run migrations (coming soon)
# cargo run --bin migrate
```

## Project Structure

```
.
├── backend/              # Rust backend API
│   ├── src/
│   ├── Cargo.toml
│   └── Dockerfile
├── analytics/            # Julia analytics engine
│   ├── src/
│   ├── Project.toml
│   └── Dockerfile
├── database/             # Database schemas and migrations
│   └── init.sql
├── frontend/             # React dashboard (coming soon)
├── docker-compose.yml    # Docker orchestration
├── .env.example          # Environment variables template
└── README.md

```

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `GET /api/v1/auth/verify` - Verify token

### GI Products
- `POST /api/v1/gi-products` - Create GI product
- `GET /api/v1/gi-products` - List GI products
- `GET /api/v1/gi-products/:id` - Get GI product by ID
- `GET /api/v1/gi-products/region/:bounds` - Query by region

### Predictions
- `POST /api/v1/predictions/roi` - Predict ROI
- `GET /api/v1/predictions/:id` - Get prediction
- `GET /api/v1/predictions/:id/explanation` - Get SHAP explanation

### Clusters
- `POST /api/v1/clusters/detect` - Detect economic clusters
- `GET /api/v1/clusters` - List clusters
- `GET /api/v1/clusters/:id` - Get cluster details

### MLA Scores
- `POST /api/v1/mla-scores/compute` - Compute MLA impact score
- `GET /api/v1/mla-scores/:constituency_id` - Get scores by constituency

### Anomalies
- `POST /api/v1/anomalies/detect` - Detect spending anomalies
- `GET /api/v1/anomalies` - List anomalies
- `GET /api/v1/anomalies/:id` - Get anomaly details

## Data Sources

- **GI Products**: [India Brand Equity Foundation - GI of India](https://www.ibef.org/giofindia)
- **MLA Development Data**: [PRS India - MLA Track](https://prsindia.org/mlatrack)

## Development

### Running Tests

```bash
# Rust tests
cd backend
cargo test

# Julia tests
cd analytics
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Code Formatting

```bash
# Rust
cargo fmt

# Julia
julia --project=. -e 'using JuliaFormatter; format(".")'
```

## Configuration

See `.env.example` for all available configuration options.

Key configurations:
- Database connection strings
- JWT secret for authentication
- API endpoints for external services
- Rate limiting settings
- Logging levels

## License

MIT

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## Support

For issues and questions, please open an issue on GitHub.
