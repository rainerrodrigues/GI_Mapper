use anyhow::{Context, Result};
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;
use tracing::{info, warn};

pub mod audit;
pub mod gi_product;
pub mod user;
pub mod cluster;
pub mod prediction;

/// Database connection pool manager
#[derive(Clone)]
pub struct Database {
    pool: PgPool,
}

impl Database {
    /// Create a new database connection pool
    ///
    /// # Arguments
    /// * `database_url` - PostgreSQL connection string
    ///
    /// # Returns
    /// * `Result<Self>` - Database instance with connection pool
    pub async fn new(database_url: &str) -> Result<Self> {
        info!("Initializing database connection pool");

        let pool = PgPoolOptions::new()
            .max_connections(20)
            .min_connections(5)
            .acquire_timeout(Duration::from_secs(10))
            .idle_timeout(Duration::from_secs(600))
            .max_lifetime(Duration::from_secs(1800))
            .connect(database_url)
            .await
            .context("Failed to create database connection pool")?;

        info!("Database connection pool created successfully");

        Ok(Self { pool })
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Perform a health check on the database connection
    ///
    /// # Returns
    /// * `Result<HealthStatus>` - Health status with connection details
    pub async fn health_check(&self) -> Result<HealthStatus> {
        // Test basic connectivity
        let start = std::time::Instant::now();
        
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .context("Database health check query failed")?;

        let query_duration = start.elapsed();

        // Check PostGIS extension
        let postgis_version: (String,) = sqlx::query_as(
            "SELECT PostGIS_Version()"
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to query PostGIS version")?;

        // Get pool statistics
        let pool_size = self.pool.size();
        let idle_connections = self.pool.num_idle();

        info!(
            "Database health check passed - PostGIS: {}, Pool: {}/{}, Query time: {:?}",
            postgis_version.0, idle_connections, pool_size, query_duration
        );

        Ok(HealthStatus {
            connected: true,
            postgis_version: postgis_version.0,
            pool_size,
            idle_connections,
            query_duration_ms: query_duration.as_millis() as u64,
        })
    }

    /// Close the database connection pool gracefully
    pub async fn close(&self) {
        info!("Closing database connection pool");
        self.pool.close().await;
    }
}

/// Database health status information
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthStatus {
    pub connected: bool,
    pub postgis_version: String,
    pub pool_size: u32,
    pub idle_connections: usize,
    pub query_duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires running database
    async fn test_database_connection() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set for tests");

        let db = Database::new(&database_url).await.unwrap();
        assert!(db.pool().size() > 0);
    }

    #[tokio::test]
    #[ignore] // Requires running database
    async fn test_health_check() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set for tests");

        let db = Database::new(&database_url).await.unwrap();
        let health = db.health_check().await.unwrap();

        assert!(health.connected);
        assert!(!health.postgis_version.is_empty());
        assert!(health.pool_size > 0);
        assert!(health.query_duration_ms < 1000); // Should be fast
    }

    #[tokio::test]
    #[ignore] // Requires running database
    async fn test_postgis_extension() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set for tests");

        let db = Database::new(&database_url).await.unwrap();
        
        // Verify PostGIS functions are available
        let result: (bool,) = sqlx::query_as(
            "SELECT ST_IsValid(ST_GeomFromText('POINT(0 0)', 4326))"
        )
        .fetch_one(db.pool())
        .await
        .unwrap();

        assert!(result.0);
    }
}
