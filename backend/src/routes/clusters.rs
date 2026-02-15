use axum::{routing::{get, post}, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Cluster routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/detect", post(detect_clusters))
        .route("/", get(list_clusters))
        .route("/:id", get(get_cluster))
        .route("/:id/explanation", get(get_explanation))
}

/// Detect clusters handler (placeholder)
async fn detect_clusters() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Detect clusters endpoint - to be implemented"
    }))
}

/// List clusters handler (placeholder)
async fn list_clusters() -> Json<serde_json::Value> {
    Json(json!({
        "message": "List clusters endpoint - to be implemented"
    }))
}

/// Get cluster by ID handler (placeholder)
async fn get_cluster() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get cluster endpoint - to be implemented"
    }))
}

/// Get cluster explanation handler (placeholder)
async fn get_explanation() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get cluster explanation endpoint - to be implemented"
    }))
}
