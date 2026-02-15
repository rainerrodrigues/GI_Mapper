use serde::{Deserialize, Serialize};
use sqlx::types::{Json, JsonValue};
use sqlx::FromRow;
use uuid::Uuid;

/// Economic cluster detected by AI spatial clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Cluster {
    pub id: Uuid,
    pub cluster_id: i32,
    #[sqlx(skip)]
    pub boundary: Vec<Vec<f64>>, // Polygon coordinates [[lon, lat], ...]
    #[sqlx(skip)]
    pub centroid: Vec<f64>, // Point coordinates [lon, lat]
    pub algorithm: String,
    pub parameters: JsonValue,
    pub member_count: i32,
    pub density: f64,
    pub economic_value: f64,
    pub quality_metrics: JsonValue,
    pub detected_at: chrono::NaiveDateTime,
    pub metadata_hash: Option<String>,
    pub blockchain_tx_id: Option<String>,
}

/// Request to detect clusters
#[derive(Debug, Deserialize)]
pub struct DetectClustersRequest {
    pub data_points: Vec<DataPoint>,
    #[serde(default = "default_algorithm")]
    pub algorithm: String, // "dbscan", "kmeans", or "auto"
    #[serde(default)]
    pub parameters: Option<ClusterParameters>,
}

fn default_algorithm() -> String {
    "auto".to_string()
}

/// Data point for clustering
#[derive(Debug, Serialize, Deserialize)]
pub struct DataPoint {
    pub id: String,
    pub latitude: f64,
    pub longitude: f64,
    #[serde(default)]
    pub value: f64, // Economic value or other attribute
}

/// Optional clustering parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterParameters {
    // DBSCAN parameters
    pub epsilon: Option<f64>,
    pub min_points: Option<i32>,
    
    // K-Means parameters
    pub k: Option<i32>,
    pub max_k: Option<i32>,
}

/// Response from cluster detection
#[derive(Debug, Serialize)]
pub struct DetectClustersResponse {
    pub clusters: Vec<ClusterInfo>,
    pub algorithm_used: String,
    pub parameters_used: JsonValue,
    pub quality_metrics: QualityMetrics,
    pub model_version: String,
}

/// Cluster information in response
#[derive(Debug, Serialize)]
pub struct ClusterInfo {
    pub id: Uuid,
    pub cluster_id: i32,
    pub boundary: Vec<Vec<f64>>,
    pub centroid: Vec<f64>,
    pub member_count: i32,
    pub density: f64,
    pub economic_value: f64,
    pub member_ids: Vec<String>,
}

/// Clustering quality metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub silhouette_score: f64,
    pub davies_bouldin_index: f64,
    pub calinski_harabasz_score: f64,
}

/// Cluster list response
#[derive(Debug, Serialize)]
pub struct ClusterListResponse {
    pub clusters: Vec<ClusterSummary>,
    pub total: usize,
}

/// Cluster summary for list view
#[derive(Debug, Serialize)]
pub struct ClusterSummary {
    pub id: Uuid,
    pub cluster_id: i32,
    pub centroid: Vec<f64>,
    pub member_count: i32,
    pub algorithm: String,
    pub detected_at: chrono::NaiveDateTime,
}
