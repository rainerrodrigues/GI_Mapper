# Database Schema and Migrations

This directory contains the PostGIS database schema and migration scripts for the AI-Powered Blockchain GIS Platform.

## Overview

The database uses PostgreSQL with PostGIS extension for spatial data storage and queries. All geographic data is stored in WGS84 coordinate system (SRID 4326).

## Schema Structure

### Core Tables

1. **gi_products** - Geographical Indication products with spatial coordinates
2. **economic_clusters** - AI-detected economic clusters with spatial boundaries
3. **roi_predictions** - Investment ROI predictions from ensemble models
4. **mla_scores** - MLA development impact scores with constituency boundaries
5. **anomalies** - Detected anomalies in development spending
6. **forecasts** - Time series forecasts for development indicators
7. **risk_assessments** - Multi-dimensional risk assessments for projects
8. **model_performance** - AI model performance tracking and KPIs
9. **users** - User authentication and authorization
10. **audit_log** - Comprehensive audit trail of system operations

### Spatial Indexes

All tables with geometry columns have GIST spatial indexes for optimized spatial queries:
- `gi_products.location` - Point geometries for GI products
- `economic_clusters.boundary` - Polygon geometries for cluster boundaries
- `economic_clusters.centroid` - Point geometries for cluster centers
- `roi_predictions.location` - Point geometries for prediction locations
- `mla_scores.constituency_boundary` - Polygon geometries for constituencies
- `anomalies.location` - Point geometries for anomaly locations
- `risk_assessments.location` - Point geometries for project locations

### Foreign Key Constraints

- `audit_log.user_id` → `users.id` - Links audit entries to users

### Check Constraints

- `anomalies.severity` - Must be one of: 'low', 'medium', 'high', 'critical'
- `risk_assessments.risk_level` - Must be one of: 'low', 'medium', 'high', 'critical'
- `users.role` - Must be one of: 'administrator', 'analyst', 'viewer'

## Migrations

### Migration 001: Create Schema

**File:** `migrations/001_create_schema.sql`

**Purpose:** Creates all tables, indexes, constraints, views, and helper functions

**Features:**
- All 10 core tables with appropriate data types
- Spatial indexes on all geometry columns
- Additional indexes for common query patterns
- Check constraints for enum-like fields
- Foreign key constraints for referential integrity
- Helper function for validating Indian geographic boundaries
- Trigger for automatic `updated_at` timestamp updates
- Views for common queries (high severity anomalies, recent model performance)

**Requirements Addressed:**
- 11.1: PostgreSQL with PostGIS extension
- 11.2: Spatial indexes for query optimization
- 11.3: Standard PostGIS operations support
- 11.4: WGS84 coordinate system (SRID 4326)
- 11.5: Geometry validation

## Running Migrations

### Using Docker Compose

Migrations are automatically applied when the database container starts:

```bash
docker-compose up -d postgres
```

The `init.sql` script runs all migrations in the `migrations/` directory.

### Manual Migration

To run migrations manually:

```bash
psql -h localhost -U gis_user -d gis_platform -f database/migrations/001_create_schema.sql
```

## Validation Functions

### validate_indian_boundaries(geom GEOMETRY)

Validates that coordinates fall within Indian geographic boundaries (approximately 6.5°N to 35.5°N, 68°E to 97.5°E).

**Usage:**
```sql
SELECT validate_indian_boundaries(ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326));
-- Returns: true (New Delhi coordinates)
```

## Views

### high_severity_anomalies

Shows anomalies with 'high' or 'critical' severity, ordered by anomaly score.

**Usage:**
```sql
SELECT * FROM high_severity_anomalies LIMIT 10;
```

### recent_model_performance

Shows model performance metrics from the last 30 days.

**Usage:**
```sql
SELECT * FROM recent_model_performance WHERE model_name = 'roi_predictor';
```

## Common Spatial Queries

### Find GI products within a region

```sql
SELECT * FROM gi_products
WHERE ST_Contains(
    ST_MakeEnvelope(77.0, 28.0, 78.0, 29.0, 4326),
    location
);
```

### Find clusters intersecting a boundary

```sql
SELECT * FROM economic_clusters
WHERE ST_Intersects(
    boundary,
    ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)
);
```

### Calculate distance between predictions

```sql
SELECT 
    p1.id,
    p2.id,
    ST_Distance(p1.location::geography, p2.location::geography) as distance_meters
FROM roi_predictions p1
CROSS JOIN roi_predictions p2
WHERE p1.id < p2.id
ORDER BY distance_meters
LIMIT 10;
```

## Backup and Restore

### Backup

```bash
pg_dump -h localhost -U gis_user -d gis_platform -F c -f backup.dump
```

### Restore

```bash
pg_restore -h localhost -U gis_user -d gis_platform -c backup.dump
```

## Performance Considerations

1. **Spatial Indexes**: All geometry columns have GIST indexes for fast spatial queries
2. **Composite Indexes**: Time-based queries use composite indexes (e.g., `model_name, measured_at DESC`)
3. **JSONB**: Used for flexible schema fields with GIN indexes where needed
4. **Partitioning**: Consider partitioning large tables (audit_log, model_performance) by time range in production

## Security

1. **Password Hashing**: User passwords are stored as bcrypt hashes
2. **Audit Trail**: All operations logged in audit_log table
3. **Role-Based Access**: Users have roles (administrator, analyst, viewer)
4. **Blockchain Verification**: Critical data hashes stored for integrity verification

## Monitoring

Key metrics to monitor:
- Table sizes: `SELECT pg_size_pretty(pg_total_relation_size('table_name'));`
- Index usage: `SELECT * FROM pg_stat_user_indexes;`
- Slow queries: Enable `pg_stat_statements` extension
- Spatial query performance: Use `EXPLAIN ANALYZE` for optimization

## References

- [PostGIS Documentation](https://postgis.net/documentation/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Spatial Indexing](https://postgis.net/workshops/postgis-intro/indexing.html)
