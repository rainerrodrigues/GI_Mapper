use axum::{
    routing::get,
    Router,
    Json,
    extract::State,
};
use serde_json::json;

use crate::db::Database;

pub mod auth;
pub mod gi_products;
pub mod predictions;
pub mod clusters;
pub mod mla_scores;
pub mod anomalies;
pub mod forecasts;
pub mod risk;
pub mod blockchain;
pub mod export;

/// Create the main application router with all routes
pub fn create_router(database: Database) -> Router {
    Router::new()
        // Health check endpoint
        .route("/health", get(health_check))
        
        // Database health check endpoint
        .route("/health/db", get(db_health_check))
        
        // API v1 routes
        .nest("/api/v1", api_v1_routes())
        .with_state(database)
}

/// API v1 routes
fn api_v1_routes() -> Router<Database> {
    Router::new()
        // Authentication routes
        .nest("/auth", auth::routes())
        
        // GI Products routes
        .nest("/gi-products", gi_products::routes())
        
        // Prediction routes
        .nest("/predictions", predictions::routes())
        
        // Cluster routes
        .nest("/clusters", clusters::routes())
        
        // MLA scores routes
        .nest("/mla-scores", mla_scores::routes())
        
        // Anomaly routes
        .nest("/anomalies", anomalies::routes())
        
        // Forecast routes
        .nest("/forecasts", forecasts::routes())
        
        // Risk assessment routes
        .nest("/risk", risk::routes())
        
        // Blockchain verification routes
        .nest("/blockchain", blockchain::routes())
        
        // Export routes
        .nest("/export", export::routes())
}

/// Health check handler
async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "service": "AI-Powered Blockchain GIS Platform API",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Database health check handler
async fn db_health_check(
    State(database): State<Database>,
) -> Json<serde_json::Value> {
    match database.health_check().await {
        Ok(health) => Json(json!({
            "status": "healthy",
            "database": health
        })),
        Err(e) => Json(json!({
            "status": "unhealthy",
            "error": e.to_string()
        })),
    }
}
