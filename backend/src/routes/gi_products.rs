use axum::{routing::{get, post}, Router, Json};
use serde_json::json;

use crate::db::Database;

/// GI Products routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/", post(create).get(list))
        .route("/:id", get(get_by_id))
        .route("/region/:bounds", get(get_by_region))
}

/// Create GI product handler (placeholder)
async fn create() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Create GI product endpoint - to be implemented"
    }))
}

/// List GI products handler (placeholder)
async fn list() -> Json<serde_json::Value> {
    Json(json!({
        "message": "List GI products endpoint - to be implemented"
    }))
}

/// Get GI product by ID handler (placeholder)
async fn get_by_id() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get GI product by ID endpoint - to be implemented"
    }))
}

/// Get GI products by region handler (placeholder)
async fn get_by_region() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get GI products by region endpoint - to be implemented"
    }))
}
