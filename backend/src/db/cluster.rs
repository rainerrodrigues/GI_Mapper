use sqlx::{PgPool, Row};
use uuid::Uuid;
use serde_json::json;

use crate::error::{AppError, Result};
use crate::models::{Cluster, ClusterSummary};

/// Store a cluster in the database
pub async fn create_cluster(
    pool: &PgPool,
    cluster_id: i32,
    boundary: &[Vec<f64>],
    centroid: &[f64],
    algorithm: &str,
    parameters: &serde_json::Value,
    member_count: i32,
    density: f64,
    economic_value: f64,
    quality_metrics: &serde_json::Value,
    metadata_hash: Option<&str>,
    blockchain_tx_id: Option<&str>,
) -> Result<Uuid> {
    // Convert boundary to PostGIS polygon format
    let boundary_wkt = polygon_to_wkt(boundary);
    let centroid_wkt = format!("POINT({} {})", centroid[0], centroid[1]);
    
    let row = sqlx::query(
        r#"
        INSERT INTO economic_clusters (
            cluster_id, boundary, centroid, algorithm, parameters,
            member_count, density, economic_value, quality_metrics,
            metadata_hash, blockchain_tx_id
        )
        VALUES (
            $1, ST_GeomFromText($2, 4326), ST_GeomFromText($3, 4326),
            $4, $5, $6, $7, $8, $9, $10, $11
        )
        RETURNING id
        "#,
    )
    .bind(cluster_id)
    .bind(&boundary_wkt)
    .bind(&centroid_wkt)
    .bind(algorithm)
    .bind(parameters)
    .bind(member_count)
    .bind(density)
    .bind(economic_value)
    .bind(quality_metrics)
    .bind(metadata_hash)
    .bind(blockchain_tx_id)
    .fetch_one(pool)
    .await
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    
    let id: Uuid = row.try_get("id")
        .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    
    Ok(id)
}

/// Get a cluster by ID
pub async fn get_cluster_by_id(pool: &PgPool, id: Uuid) -> Result<Cluster> {
    let row = sqlx::query(
        r#"
        SELECT 
            id, cluster_id,
            ST_AsText(boundary) as boundary_wkt,
            ST_AsText(centroid) as centroid_wkt,
            algorithm, parameters, member_count, density, economic_value,
            quality_metrics, detected_at, metadata_hash, blockchain_tx_id
        FROM economic_clusters
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_one(pool)
    .await
    .map_err(|e| match e {
        sqlx::Error::RowNotFound => AppError::NotFound("Cluster not found".to_string()),
        _ => AppError::DatabaseError(e.to_string()),
    })?;
    
    let boundary_wkt: String = row.try_get("boundary_wkt")
        .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    let centroid_wkt: String = row.try_get("centroid_wkt")
        .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    
    let boundary = parse_polygon_wkt(&boundary_wkt)?;
    let centroid = parse_point_wkt(&centroid_wkt)?;
    
    Ok(Cluster {
        id: row.try_get("id").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        cluster_id: row.try_get("cluster_id").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        boundary,
        centroid,
        algorithm: row.try_get("algorithm").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        parameters: row.try_get("parameters").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        member_count: row.try_get("member_count").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        density: row.try_get("density").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        economic_value: row.try_get("economic_value").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        quality_metrics: row.try_get("quality_metrics").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        detected_at: row.try_get("detected_at").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        metadata_hash: row.try_get("metadata_hash").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        blockchain_tx_id: row.try_get("blockchain_tx_id").map_err(|e| AppError::DatabaseError(e.to_string()))?,
    })
}

/// List all clusters
pub async fn list_clusters(pool: &PgPool, limit: i64, offset: i64) -> Result<Vec<ClusterSummary>> {
    let rows = sqlx::query(
        r#"
        SELECT 
            id, cluster_id,
            ST_AsText(centroid) as centroid_wkt,
            member_count, algorithm, detected_at
        FROM economic_clusters
        ORDER BY detected_at DESC
        LIMIT $1 OFFSET $2
        "#,
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await
    .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    
    let mut clusters = Vec::new();
    for row in rows {
        let centroid_wkt: String = row.try_get("centroid_wkt")
            .map_err(|e| AppError::DatabaseError(e.to_string()))?;
        let centroid = parse_point_wkt(&centroid_wkt)?;
        
        clusters.push(ClusterSummary {
            id: row.try_get("id").map_err(|e| AppError::DatabaseError(e.to_string()))?,
            cluster_id: row.try_get("cluster_id").map_err(|e| AppError::DatabaseError(e.to_string()))?,
            centroid,
            member_count: row.try_get("member_count").map_err(|e| AppError::DatabaseError(e.to_string()))?,
            algorithm: row.try_get("algorithm").map_err(|e| AppError::DatabaseError(e.to_string()))?,
            detected_at: row.try_get("detected_at").map_err(|e| AppError::DatabaseError(e.to_string()))?,
        });
    }
    
    Ok(clusters)
}

/// Count total clusters
pub async fn count_clusters(pool: &PgPool) -> Result<i64> {
    let row = sqlx::query("SELECT COUNT(*) as count FROM economic_clusters")
        .fetch_one(pool)
        .await
        .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    
    let count: i64 = row.try_get("count")
        .map_err(|e| AppError::DatabaseError(e.to_string()))?;
    
    Ok(count)
}

// Helper functions

/// Convert polygon coordinates to WKT format
fn polygon_to_wkt(coords: &[Vec<f64>]) -> String {
    let points: Vec<String> = coords
        .iter()
        .map(|p| format!("{} {}", p[0], p[1]))
        .collect();
    
    format!("POLYGON(({}))", points.join(", "))
}

/// Parse WKT polygon to coordinates
fn parse_polygon_wkt(wkt: &str) -> Result<Vec<Vec<f64>>> {
    // Example: "POLYGON((lon1 lat1, lon2 lat2, ...))"
    let coords_str = wkt
        .trim_start_matches("POLYGON((")
        .trim_end_matches("))")
        .trim();
    
    let coords: Result<Vec<Vec<f64>>> = coords_str
        .split(", ")
        .map(|point| {
            let parts: Vec<&str> = point.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(AppError::ValidationError("Invalid polygon format".to_string()));
            }
            
            let lon = parts[0].parse::<f64>()
                .map_err(|_| AppError::ValidationError("Invalid longitude".to_string()))?;
            let lat = parts[1].parse::<f64>()
                .map_err(|_| AppError::ValidationError("Invalid latitude".to_string()))?;
            
            Ok(vec![lon, lat])
        })
        .collect();
    
    coords
}

/// Parse WKT point to coordinates
fn parse_point_wkt(wkt: &str) -> Result<Vec<f64>> {
    // Example: "POINT(lon lat)"
    let coords_str = wkt
        .trim_start_matches("POINT(")
        .trim_end_matches(")")
        .trim();
    
    let parts: Vec<&str> = coords_str.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(AppError::ValidationError("Invalid point format".to_string()));
    }
    
    let lon = parts[0].parse::<f64>()
        .map_err(|_| AppError::ValidationError("Invalid longitude".to_string()))?;
    let lat = parts[1].parse::<f64>()
        .map_err(|_| AppError::ValidationError("Invalid latitude".to_string()))?;
    
    Ok(vec![lon, lat])
}
