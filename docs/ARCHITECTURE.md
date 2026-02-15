# Architecture Overview

## System Components

### 1. Backend API (Rust)
- **Framework**: Axum
- **Responsibilities**: HTTP API, authentication, orchestration
- **Port**: 8000

### 2. Analytics Engine (Julia)
- **Framework**: gRPC
- **Responsibilities**: AI/ML computations, spatial analytics
- **Port**: 50051

### 3. PostgreSQL + PostGIS
- **Purpose**: Geospatial data storage
- **Port**: 5432

### 4. Redis
- **Purpose**: Caching and rate limiting
- **Port**: 6379

### 5. IPFS
- **Purpose**: Distributed evidence storage
- **Ports**: 4001 (P2P), 5001 (API), 8080 (Gateway)

### 6. Substrate Blockchain
- **Purpose**: Immutable audit trails
- **Ports**: 9944 (WebSocket), 9933 (RPC), 30333 (P2P)

## Data Flow

1. **User Request** → Backend API (Rust)
2. **Backend** → Analytics Engine (Julia) via gRPC
3. **Analytics** → PostGIS for data retrieval
4. **Analytics** → ML/AI computation
5. **Analytics** → Backend with results
6. **Backend** → PostGIS to store results
7. **Backend** → Blockchain to store hash
8. **Backend** → User with response

## Communication Patterns

- **Frontend ↔ Backend**: REST API over HTTPS
- **Backend ↔ Analytics**: gRPC with Protocol Buffers
- **Backend ↔ Database**: SQLx connection pool
- **Backend ↔ Blockchain**: Substrate RPC
- **Backend ↔ IPFS**: HTTP API

## Security Layers

1. **Transport**: TLS 1.3 encryption
2. **Authentication**: JWT tokens
3. **Authorization**: Role-based access control
4. **Data**: AES-256 encryption at rest
5. **Audit**: Blockchain verification
