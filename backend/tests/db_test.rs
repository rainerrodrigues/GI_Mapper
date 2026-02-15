use backend::db::{Database, HealthStatus};

/// Test database connection pool creation
#[tokio::test]
#[ignore] // Requires running database
async fn test_database_connection_pool() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await;
    assert!(db.is_ok(), "Failed to create database connection pool");

    let db = db.unwrap();
    assert!(db.pool().size() > 0, "Pool should have connections");
}

/// Test database health check
#[tokio::test]
#[ignore] // Requires running database
async fn test_database_health_check() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();
    let health = db.health_check().await;

    assert!(health.is_ok(), "Health check should succeed");

    let health = health.unwrap();
    assert!(health.connected, "Database should be connected");
    assert!(!health.postgis_version.is_empty(), "PostGIS version should not be empty");
    assert!(health.pool_size > 0, "Pool size should be greater than 0");
    assert!(health.query_duration_ms < 1000, "Query should be fast (< 1 second)");
}

/// Test PostGIS extension availability
#[tokio::test]
#[ignore] // Requires running database
async fn test_postgis_extension() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();

    // Test PostGIS function
    let result: Result<(bool,), sqlx::Error> = sqlx::query_as(
        "SELECT ST_IsValid(ST_GeomFromText('POINT(0 0)', 4326))"
    )
    .fetch_one(db.pool())
    .await;

    assert!(result.is_ok(), "PostGIS function should work");
    assert!(result.unwrap().0, "Point geometry should be valid");
}

/// Test spatial query with PostGIS
#[tokio::test]
#[ignore] // Requires running database
async fn test_spatial_query() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();

    // Test ST_Distance function
    let result: Result<(f64,), sqlx::Error> = sqlx::query_as(
        "SELECT ST_Distance(
            ST_GeomFromText('POINT(0 0)', 4326),
            ST_GeomFromText('POINT(1 1)', 4326)
        )"
    )
    .fetch_one(db.pool())
    .await;

    assert!(result.is_ok(), "Spatial distance query should work");
    let distance = result.unwrap().0;
    assert!(distance > 0.0, "Distance should be positive");
}

/// Test connection pool limits
#[tokio::test]
#[ignore] // Requires running database
async fn test_connection_pool_limits() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();

    // Pool should have configured limits
    assert!(db.pool().size() <= 20, "Pool size should not exceed max_connections");
    assert!(db.pool().size() >= 5, "Pool size should meet min_connections");
}

/// Test concurrent database access
#[tokio::test]
#[ignore] // Requires running database
async fn test_concurrent_access() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();

    // Spawn multiple concurrent queries
    let mut handles = vec![];
    for _ in 0..10 {
        let db_clone = db.clone();
        let handle = tokio::spawn(async move {
            sqlx::query("SELECT 1")
                .execute(db_clone.pool())
                .await
        });
        handles.push(handle);
    }

    // All queries should succeed
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent query should succeed");
    }
}

/// Test database health check serialization
#[test]
fn test_health_status_serialization() {
    let health = HealthStatus {
        connected: true,
        postgis_version: "3.4.0".to_string(),
        pool_size: 10,
        idle_connections: 5,
        query_duration_ms: 15,
    };

    let json = serde_json::to_string(&health).unwrap();
    assert!(json.contains("connected"));
    assert!(json.contains("postgis_version"));
    assert!(json.contains("3.4.0"));
}
