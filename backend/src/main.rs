use anyhow::Result;
use dotenv::dotenv;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
pub mod db;
mod error;
mod middleware;
mod models;
mod routes;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "backend=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = config::Config::from_env()?;
    
    tracing::info!("Starting AI-Powered Blockchain GIS Platform - Backend API");
    tracing::info!("Server will listen on {}", config.server_addr);

    // Initialize database connection pool
    let database = db::Database::new(&config.database_url).await?;
    
    // Perform initial health check
    match database.health_check().await {
        Ok(health) => {
            tracing::info!("Database health check passed: {:?}", health);
        }
        Err(e) => {
            tracing::error!("Database health check failed: {}", e);
            return Err(e);
        }
    }

    // Build application with routes and middleware
    let app = routes::create_router(database.clone())
        .layer(middleware::create_middleware_stack());

    // Start server
    let listener = tokio::net::TcpListener::bind(&config.server_addr).await?;
    tracing::info!("Server listening on {}", config.server_addr);
    
    axum::serve(listener, app).await?;

    Ok(())
}
