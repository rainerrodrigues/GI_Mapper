use axum::{routing::{get, post}, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Forecast routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/generate", post(generate_forecast))
        .route("/:id", get(get_forecast))
}

/// Generate forecast handler (placeholder)
async fn generate_forecast() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Generate forecast endpoint - to be implemented"
    }))
}

/// Get forecast by ID handler (placeholder)
async fn get_forecast() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get forecast endpoint - to be implemented"
    }))
}
