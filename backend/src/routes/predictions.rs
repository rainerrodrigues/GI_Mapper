use axum::{routing::{get, post}, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Prediction routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/roi", post(predict_roi))
        .route("/:id", get(get_prediction))
        .route("/:id/explanation", get(get_explanation))
}

/// Predict ROI handler (placeholder)
async fn predict_roi() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Predict ROI endpoint - to be implemented"
    }))
}

/// Get prediction by ID handler (placeholder)
async fn get_prediction() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get prediction endpoint - to be implemented"
    }))
}

/// Get prediction explanation handler (placeholder)
async fn get_explanation() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get prediction explanation endpoint - to be implemented"
    }))
}
