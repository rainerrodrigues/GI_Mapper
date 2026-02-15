use axum::{routing::{get, post}, Router, Json};
use serde_json::json;

use crate::db::Database;

/// MLA scores routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/compute", post(compute_score))
        .route("/", get(list_scores))
        .route("/:constituency_id", get(get_by_constituency))
        .route("/:id/explanation", get(get_explanation))
}

/// Compute MLA score handler (placeholder)
async fn compute_score() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Compute MLA score endpoint - to be implemented"
    }))
}

/// List MLA scores handler (placeholder)
async fn list_scores() -> Json<serde_json::Value> {
    Json(json!({
        "message": "List MLA scores endpoint - to be implemented"
    }))
}

/// Get MLA score by constituency handler (placeholder)
async fn get_by_constituency() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get MLA score by constituency endpoint - to be implemented"
    }))
}

/// Get MLA score explanation handler (placeholder)
async fn get_explanation() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get MLA score explanation endpoint - to be implemented"
    }))
}
