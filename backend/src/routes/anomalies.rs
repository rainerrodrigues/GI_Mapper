use axum::{routing::{get, post}, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Anomaly routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/detect", post(detect_anomalies))
        .route("/", get(list_anomalies))
        .route("/:id", get(get_anomaly))
        .route("/:id/explanation", get(get_explanation))
}

/// Detect anomalies handler (placeholder)
async fn detect_anomalies() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Detect anomalies endpoint - to be implemented"
    }))
}

/// List anomalies handler (placeholder)
async fn list_anomalies() -> Json<serde_json::Value> {
    Json(json!({
        "message": "List anomalies endpoint - to be implemented"
    }))
}

/// Get anomaly by ID handler (placeholder)
async fn get_anomaly() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get anomaly endpoint - to be implemented"
    }))
}

/// Get anomaly explanation handler (placeholder)
async fn get_explanation() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get anomaly explanation endpoint - to be implemented"
    }))
}
