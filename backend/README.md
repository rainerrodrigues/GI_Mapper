# Backend API

Rust-based backend API for the AI-Powered Blockchain GIS Platform.

## Architecture

The backend is built with:
- **Axum** - Web framework for routing and HTTP handling
- **Tower** - Middleware stack for CORS, logging, and error handling
- **SQLx** - PostgreSQL/PostGIS database access
- **Redis** - Caching and rate limiting
- **Tonic** - gRPC client for Julia analytics engine
- **Subxt** - Substrate blockchain client
- **Reqwest** - IPFS HTTP client

## Structure

```
backend/
├── src/
│   ├── main.rs           # Application entry point
│   ├── config.rs         # Configuration management
│   ├── error.rs          # Error types and handling
│   ├── middleware.rs     # Middleware stack (CORS, logging)
│   └── routes/           # API route handlers
│       ├── mod.rs        # Router setup
│       ├── auth.rs       # Authentication endpoints
│       ├── gi_products.rs    # GI product CRUD
│       ├── predictions.rs    # ROI predictions
│       ├── clusters.rs       # Cluster detection
│       ├── mla_scores.rs     # MLA scoring
│       ├── anomalies.rs      # Anomaly detection
│       ├── forecasts.rs      # Time series forecasting
│       ├── risk.rs           # Risk assessment
│       ├── blockchain.rs     # Blockchain verification
│       └── export.rs         # Data export
├── tests/
│   └── integration_test.rs
├── Cargo.toml
├── Dockerfile
└── README.md
```

## API Endpoints

### Health Check
- `GET /health` - Server health status

### Authentication (v1)
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/verify` - Verify JWT token

### GI Products
- `POST /api/v1/gi-products` - Create GI product
- `GET /api/v1/gi-products` - List GI products
- `GET /api/v1/gi-products/:id` - Get GI product by ID
- `GET /api/v1/gi-products/region/:bounds` - Get products by region

### Predictions
- `POST /api/v1/predictions/roi` - Predict ROI
- `GET /api/v1/predictions/:id` - Get prediction
- `GET /api/v1/predictions/:id/explanation` - Get SHAP explanation

### Clusters
- `POST /api/v1/clusters/detect` - Detect clusters
- `GET /api/v1/clusters` - List clusters
- `GET /api/v1/clusters/:id` - Get cluster
- `GET /api/v1/clusters/:id/explanation` - Get explanation

### MLA Scores
- `POST /api/v1/mla-scores/compute` - Compute score
- `GET /api/v1/mla-scores` - List scores
- `GET /api/v1/mla-scores/:constituency_id` - Get by constituency
- `GET /api/v1/mla-scores/:id/explanation` - Get explanation

### Anomalies
- `POST /api/v1/anomalies/detect` - Detect anomalies
- `GET /api/v1/anomalies` - List anomalies
- `GET /api/v1/anomalies/:id` - Get anomaly
- `GET /api/v1/anomalies/:id/explanation` - Get explanation

### Forecasts
- `POST /api/v1/forecasts/generate` - Generate forecast
- `GET /api/v1/forecasts/:id` - Get forecast

### Risk Assessment
- `POST /api/v1/risk/assess` - Assess risk
- `GET /api/v1/risk/:id` - Get assessment

### Blockchain
- `GET /api/v1/blockchain/verify/:hash` - Verify data hash
- `GET /api/v1/blockchain/audit-trail/:entity_id` - Get audit trail

### Export
- `POST /api/v1/export/csv` - Export as CSV
- `POST /api/v1/export/geojson` - Export as GeoJSON

## Configuration

Configuration is loaded from environment variables:

- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DATABASE_URL` - PostgreSQL connection string (required)
- `REDIS_URL` - Redis connection string (default: redis://localhost:6379)
- `JWT_SECRET` - JWT signing secret (required)
- `RUST_LOG` - Logging level (default: info)

## Middleware

### CORS
- Allows requests from any origin (configurable for production)
- Supports GET, POST, PUT, DELETE, OPTIONS methods
- Allows Content-Type, Authorization, Accept headers

### Logging
- Logs all HTTP requests with timestamps
- Logs response times and status codes
- Uses structured logging with tracing

### Error Handling
- Returns structured JSON error responses
- Includes error codes and descriptive messages
- Appropriate HTTP status codes

## Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  }
}
```

Error codes:
- `INTERNAL_ERROR` - Internal server error (500)
- `BAD_REQUEST` - Invalid request (400)
- `UNAUTHORIZED` - Authentication required (401)
- `FORBIDDEN` - Insufficient permissions (403)
- `NOT_FOUND` - Resource not found (404)

## Development

### Running locally
```bash
# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run with cargo
cargo run

# Or use Docker Compose
docker-compose up backend
```

### Testing
```bash
cargo test
```

### Building
```bash
cargo build --release
```

## Next Steps

The current implementation provides the routing structure and middleware. Future tasks will implement:

1. Database connection pooling (Task 3.7)
2. Authentication and JWT handling (Task 3.2)
3. Rate limiting (Task 3.6)
4. Individual endpoint handlers with business logic
5. Integration with Julia analytics engine via gRPC
6. Blockchain client for data verification
7. IPFS client for evidence storage
