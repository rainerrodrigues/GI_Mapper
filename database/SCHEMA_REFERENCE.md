# Database Schema Quick Reference

## Table Summary

| Table | Purpose | Spatial Columns | Key Indexes |
|-------|---------|-----------------|-------------|
| `gi_products` | GI products with locations | `location` (Point) | location, region, category |
| `economic_clusters` | AI-detected clusters | `boundary` (Polygon), `centroid` (Point) | boundary, centroid, algorithm |
| `roi_predictions` | Investment ROI predictions | `location` (Point) | location, sector, model_version |
| `mla_scores` | MLA development scores | `constituency_boundary` (Polygon) | boundary, constituency_id, overall_score |
| `anomalies` | Spending anomalies | `location` (Point) | location, severity, score |
| `forecasts` | Time series forecasts | None | region, indicator, model_type |
| `risk_assessments` | Project risk assessments | `location` (Point) | location, project_id, risk_level |
| `model_performance` | Model KPI tracking | None | model_name, metric_name |
| `users` | User accounts | None | username, role |
| `audit_log` | System audit trail | None | user_id, entity_type, timestamp |

## Coordinate System

All spatial data uses **WGS84 (SRID 4326)**:
- Longitude: -180 to 180 (X coordinate)
- Latitude: -90 to 90 (Y coordinate)
- For India: approximately 68°E to 97.5°E, 6.5°N to 35.5°N

## Common Data Types

- **UUID**: Primary keys (auto-generated)
- **GEOMETRY(Point, 4326)**: Point locations
- **GEOMETRY(Polygon, 4326)**: Area boundaries
- **JSONB**: Flexible structured data (metrics, features, SHAP values)
- **DOUBLE PRECISION**: Numeric scores and measurements
- **TIMESTAMP**: Date/time with timezone support
- **TEXT[]**: Arrays of text (e.g., recommendations)

## Key Relationships

```
users (1) ──< (N) audit_log
```

## Enum-like Fields (Check Constraints)

### anomalies.severity
- `low`
- `medium`
- `high`
- `critical`

### risk_assessments.risk_level
- `low`
- `medium`
- `high`
- `critical`

### users.role
- `administrator`
- `analyst`
- `viewer`

## JSONB Field Structures

### economic_clusters.parameters
```json
{
  "algorithm": "DBSCAN",
  "epsilon": 0.5,
  "min_points": 5
}
```

### economic_clusters.quality_metrics
```json
{
  "silhouette_score": 0.75,
  "davies_bouldin_index": 0.8,
  "calinski_harabasz_score": 150.5
}
```

### roi_predictions.features
```json
{
  "sector": "agriculture",
  "investment_amount": 1000000,
  "timeframe_years": 5,
  "population_density": 500,
  "infrastructure_score": 0.7
}
```

### roi_predictions.shap_values
```json
{
  "features": [
    {"name": "sector", "value": 0.15},
    {"name": "infrastructure_score", "value": 0.12},
    {"name": "population_density", "value": 0.08}
  ]
}
```

### mla_scores.weights
```json
{
  "infrastructure": 0.25,
  "education": 0.20,
  "healthcare": 0.20,
  "employment": 0.20,
  "economic_growth": 0.15
}
```

### mla_scores.indicators
```json
{
  "infrastructure": {
    "roads_built_km": 150,
    "bridges_constructed": 5,
    "electricity_coverage_pct": 85
  },
  "education": {
    "schools_built": 10,
    "literacy_rate_pct": 75
  }
}
```

### anomalies.deviation_metrics
```json
{
  "z_score": 3.5,
  "percentile_rank": 99.5
}
```

### forecasts.forecast_values
```json
[
  {
    "date": "2025-01-01",
    "value": 1500,
    "lower_bound": 1400,
    "upper_bound": 1600
  },
  {
    "date": "2026-01-01",
    "value": 1650,
    "lower_bound": 1520,
    "upper_bound": 1780
  }
]
```

## Blockchain Integration Fields

Every major table includes:
- `metadata_hash` (VARCHAR(64)): SHA-256 hash of record metadata
- `blockchain_tx_id` (VARCHAR(128)): Substrate blockchain transaction ID

These fields enable verification of data integrity through blockchain audit trail.

## Spatial Query Examples

### Point in Polygon
```sql
-- Find which cluster contains a point
SELECT * FROM economic_clusters
WHERE ST_Contains(boundary, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326));
```

### Distance Calculation
```sql
-- Find GI products within 10km of a location
SELECT *, ST_Distance(location::geography, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)::geography) as distance_m
FROM gi_products
WHERE ST_DWithin(location::geography, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)::geography, 10000)
ORDER BY distance_m;
```

### Bounding Box Query
```sql
-- Find all predictions in a rectangular region
SELECT * FROM roi_predictions
WHERE ST_Contains(
    ST_MakeEnvelope(77.0, 28.0, 78.0, 29.0, 4326),
    location
);
```

### Intersection
```sql
-- Find clusters that intersect with a constituency
SELECT c.* FROM economic_clusters c
JOIN mla_scores m ON ST_Intersects(c.boundary, m.constituency_boundary)
WHERE m.constituency_id = 'DL-001';
```

## Performance Tips

1. **Always use spatial indexes**: Queries with `ST_Contains`, `ST_Intersects`, `ST_DWithin` automatically use GIST indexes
2. **Use geography for distance**: Cast to `::geography` for accurate distance calculations in meters
3. **Limit result sets**: Use `LIMIT` and pagination for large queries
4. **Analyze query plans**: Use `EXPLAIN ANALYZE` to optimize slow queries
5. **Update statistics**: Run `ANALYZE table_name` after bulk inserts

## Maintenance Commands

```sql
-- Update table statistics
ANALYZE gi_products;

-- Rebuild spatial index
REINDEX INDEX idx_gi_products_location;

-- Check table size
SELECT pg_size_pretty(pg_total_relation_size('gi_products'));

-- Check index usage
SELECT * FROM pg_stat_user_indexes WHERE relname = 'gi_products';

-- Vacuum table
VACUUM ANALYZE gi_products;
```

## Migration History

| Version | Date | Description |
|---------|------|-------------|
| 001 | 2024 | Initial schema creation with all tables, indexes, and constraints |
