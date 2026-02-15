use anyhow::{Context, Result};
use std::env;
use std::net::SocketAddr;

/// Application configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub server_addr: SocketAddr,
    pub database_url: String,
    pub redis_url: String,
    pub jwt_secret: String,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = env::var("PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse::<u16>()
            .context("Invalid PORT value")?;
        
        let server_addr = format!("{}:{}", host, port)
            .parse()
            .context("Invalid server address")?;

        let database_url = env::var("DATABASE_URL")
            .context("DATABASE_URL must be set")?;

        let redis_url = env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let jwt_secret = env::var("JWT_SECRET")
            .context("JWT_SECRET must be set")?;

        Ok(Self {
            server_addr,
            database_url,
            redis_url,
            jwt_secret,
        })
    }
}
