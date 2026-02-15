use axum::{routing::post, Router, Json};
use serde_json::json;

use crate::db::Database;

/// Export routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/csv", post(export_csv))
        .route("/geojson", post(export_geojson))
}

/// Export CSV handler (placeholder)
async fn export_csv() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Export CSV endpoint - to be implemented"
    }))
}

/// Export GeoJSON handler (placeholder)
async fn export_geojson() -> Json<serde_json::Value> {
    Json(json!({
        "message": "Export GeoJSON endpoint - to be implemented"
    }))
}
