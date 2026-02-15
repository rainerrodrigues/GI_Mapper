use axum::{routing::post, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Authentication routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/login", post(login))
        .route("/logout", post(logout))
        .route("/verify", post(verify))
}

/// Login handler (placeholder)
async fn login() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Login endpoint - to be implemented"
    }))
}

/// Logout handler (placeholder)
async fn logout() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Logout endpoint - to be implemented"
    }))
}

/// Verify token handler (placeholder)
async fn verify() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Verify endpoint - to be implemented"
    }))
}
