use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{error, info};
use uuid::Uuid;

use crate::db::Database;
use crate::error::{AppError, Result};
use crate::models::{
    ClusterInfo, ClusterListResponse, DetectClustersRequest, DetectClustersResponse,
    QualityMetrics,
};

/// Cluster routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/detect", post(detect_clusters))
        .route("/", get(list_clusters))
        .route("/:id", get(get_cluster))
        .route("/:id/explanation", get(get_explanation))
}

/// Detect clusters handler
/// POST /api/v1/clusters/detect
async fn detect_clusters(
    State(database): State<Database>,
    Json(request): Json<DetectClustersRequest>,
) -> Result<Json<DetectClustersResponse>> {
    info!(
        "Cluster detection request received with {} data points, algorithm: {}",
        request.data_points.len(),
        request.algorithm
    );

    // Validate input
    if request.data_points.is_empty() {
        return Err(AppError::ValidationError(
            "At least one data point is required".to_string(),
        ));
    }

    // Call Julia analytics engine
    let analytics_result = call_julia_clustering(&request).await?;

    // Store clusters in database
    let mut cluster_infos = Vec::new();
    let mut member_map: std::collections::HashMap<i32, Vec<String>> =
        std::collections::HashMap::new();

    // Build member map from request data points
    // In a real implementation, the Julia service would return member assignments
    // For now, we'll distribute points evenly across clusters for demo
    for (idx, point) in request.data_points.iter().enumerate() {
        let cluster_id = (idx % analytics_result.clusters.len().max(1)) as i32 + 1;
        member_map
            .entry(cluster_id)
            .or_insert_with(Vec::new)
            .push(point.id.clone());
    }

    for cluster in &analytics_result.clusters {
        let cluster_id = cluster.cluster_id;
        let boundary = &cluster.boundary;
        let centroid = &cluster.centroid;
        let member_count = cluster.member_count;
        let density = cluster.density;
        let economic_value = cluster.economic_value;

        // Compute metadata hash
        let metadata = json!({
            "cluster_id": cluster_id,
            "boundary": boundary,
            "centroid": centroid,
            "algorithm": &analytics_result.algorithm_used,
            "parameters": &analytics_result.parameters_used,
            "member_count": member_count,
            "density": density,
            "economic_value": economic_value,
            "quality_metrics": &analytics_result.quality_metrics,
        });

        let metadata_str = serde_json::to_string(&metadata)
            .map_err(|e| AppError::InternalError(format!("Failed to serialize metadata: {}", e)))?;

        let mut hasher = Sha256::new();
        hasher.update(metadata_str.as_bytes());
        let metadata_hash = format!("{:x}", hasher.finalize());

        // Mock blockchain submission (for demo)
        let blockchain_tx_id = mock_blockchain_submit(&metadata_hash).await?;

        info!(
            "Submitting cluster {} to blockchain (mocked): {}",
            cluster_id, blockchain_tx_id
        );

        // Store in database
        let db_id = crate::db::cluster::create_cluster(
            database.pool(),
            cluster_id,
            boundary,
            centroid,
            &analytics_result.algorithm_used,
            &analytics_result.parameters_used,
            member_count,
            density,
            economic_value,
            &json!(analytics_result.quality_metrics),
            Some(&metadata_hash),
            Some(&blockchain_tx_id),
        )
        .await?;

        let member_ids = member_map.get(&cluster_id).cloned().unwrap_or_default();

        cluster_infos.push(ClusterInfo {
            id: db_id,
            cluster_id,
            boundary: boundary.clone(),
            centroid: centroid.clone(),
            member_count,
            density,
            economic_value,
            member_ids,
        });
    }

    info!(
        "Cluster detection complete: {} clusters detected using {}",
        cluster_infos.len(),
        analytics_result.algorithm_used
    );

    Ok(Json(DetectClustersResponse {
        clusters: cluster_infos,
        algorithm_used: analytics_result.algorithm_used,
        parameters_used: analytics_result.parameters_used,
        quality_metrics: analytics_result.quality_metrics,
        model_version: analytics_result.model_version,
    }))
}

/// List clusters handler
/// GET /api/v1/clusters
async fn list_clusters(State(database): State<Database>) -> Result<Json<ClusterListResponse>> {
    let clusters = crate::db::cluster::list_clusters(database.pool(), 100, 0).await?;
    let total = crate::db::cluster::count_clusters(database.pool()).await? as usize;

    Ok(Json(ClusterListResponse { clusters, total }))
}

/// Get cluster by ID handler
/// GET /api/v1/clusters/:id
async fn get_cluster(
    State(database): State<Database>,
    Path(id): Path<Uuid>,
) -> Result<Json<crate::models::Cluster>> {
    let cluster = crate::db::cluster::get_cluster_by_id(database.pool(), id).await?;
    Ok(Json(cluster))
}

/// Get cluster explanation handler (placeholder)
/// GET /api/v1/clusters/:id/explanation
async fn get_explanation(Path(id): Path<Uuid>) -> Json<serde_json::Value> {
    // TODO: Implement SHAP explanations in future task
    Json(json!({
        "cluster_id": id,
        "explanation": "SHAP explanations not yet implemented",
        "feature_importance": []
    }))
}

// Helper functions

/// Call Julia analytics engine for clustering
/// In production, this would use gRPC. For demo, we'll use HTTP/JSON
async fn call_julia_clustering(
    request: &DetectClustersRequest,
) -> Result<JuliaClusterResponse> {
    // For now, simulate the Julia service response
    // In production, this would make an actual HTTP/gRPC call to Julia service
    
    info!("Calling Julia analytics engine (simulated for demo)");
    
    // Simulate clustering logic
    let n_points = request.data_points.len();
    let n_clusters = (n_points / 10).max(2).min(5); // Simple heuristic
    
    let mut clusters = Vec::new();
    
    for i in 0..n_clusters {
        // Calculate simple statistics for demo
        let points_in_cluster = n_points / n_clusters;
        let start_idx = i * points_in_cluster;
        let end_idx = ((i + 1) * points_in_cluster).min(n_points);
        
        let cluster_points: Vec<_> = request.data_points[start_idx..end_idx].to_vec();
        
        if cluster_points.is_empty() {
            continue;
        }
        
        // Calculate centroid
        let avg_lon: f64 = cluster_points.iter().map(|p| p.longitude).sum::<f64>() / cluster_points.len() as f64;
        let avg_lat: f64 = cluster_points.iter().map(|p| p.latitude).sum::<f64>() / cluster_points.len() as f64;
        
        // Calculate bounding box as boundary
        let min_lon = cluster_points.iter().map(|p| p.longitude).fold(f64::INFINITY, f64::min);
        let max_lon = cluster_points.iter().map(|p| p.longitude).fold(f64::NEG_INFINITY, f64::max);
        let min_lat = cluster_points.iter().map(|p| p.latitude).fold(f64::INFINITY, f64::min);
        let max_lat = cluster_points.iter().map(|p| p.latitude).fold(f64::NEG_INFINITY, f64::max);
        
        let boundary = vec![
            vec![min_lon, min_lat],
            vec![max_lon, min_lat],
            vec![max_lon, max_lat],
            vec![min_lon, max_lat],
            vec![min_lon, min_lat], // Close polygon
        ];
        
        let economic_value: f64 = cluster_points.iter().map(|p| p.value).sum();
        let area = (max_lon - min_lon) * (max_lat - min_lat);
        let density = if area > 0.0 { cluster_points.len() as f64 / area } else { 0.0 };
        
        clusters.push(JuliaCluster {
            cluster_id: (i + 1) as i32,
            boundary,
            centroid: vec![avg_lon, avg_lat],
            member_count: cluster_points.len() as i32,
            density,
            economic_value,
        });
    }
    
    // Simulate quality metrics
    let quality_metrics = QualityMetrics {
        silhouette_score: 0.65,
        davies_bouldin_index: 0.85,
        calinski_harabasz_score: 150.0,
    };
    
    Ok(JuliaClusterResponse {
        clusters,
        algorithm_used: request.algorithm.clone(),
        parameters_used: json!({}),
        quality_metrics,
        model_version: "0.2.0".to_string(),
    })
}

/// Mock blockchain submission
/// In production, this would submit to actual Substrate blockchain
async fn mock_blockchain_submit(metadata_hash: &str) -> Result<String> {
    // Simulate blockchain transaction
    let tx_id = format!("0x{}", &metadata_hash[..16]);
    
    info!("Mock blockchain submission - would submit hash: {}", metadata_hash);
    info!("Mock blockchain transaction ID: {}", tx_id);
    
    Ok(tx_id)
}

// Response structures from Julia service

#[derive(Debug, serde::Deserialize)]
struct JuliaClusterResponse {
    clusters: Vec<JuliaCluster>,
    algorithm_used: String,
    parameters_used: serde_json::Value,
    quality_metrics: QualityMetrics,
    model_version: String,
}

#[derive(Debug, serde::Deserialize)]
struct JuliaCluster {
    cluster_id: i32,
    boundary: Vec<Vec<f64>>,
    centroid: Vec<f64>,
    member_count: i32,
    density: f64,
    economic_value: f64,
}
