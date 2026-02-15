use axum::{routing::get, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Blockchain verification routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/verify/:hash", get(verify_hash))
        .route("/audit-trail/:entity_id", get(get_audit_trail))
}

/// Verify hash handler (placeholder)
async fn verify_hash() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Verify hash endpoint - to be implemented"
    }))
}

/// Get audit trail handler (placeholder)
async fn get_audit_trail() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Get audit trail endpoint - to be implemented"
    }))
}
