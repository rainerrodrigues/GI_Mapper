use backend::db::gi_product::*;
use backend::models::{CreateGIProductRequest, GeographicBounds, UpdateGIProductRequest};
use sqlx::PgPool;

/// Helper function to create a test product
async fn create_test_product(pool: &PgPool, name: &str) -> uuid::Uuid {
    let request = CreateGIProductRequest {
        name: name.to_string(),
        description: Some("Test product".to_string()),
        category: "Test".to_string(),
        latitude: 28.6139,
        longitude: 77.2090,
        region: "Delhi".to_string(),
    };
    let product = create_gi_product(pool, request).await.unwrap();
    product.id
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_create_and_get_gi_product() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create product
    let request = CreateGIProductRequest {
        name: "Darjeeling Tea".to_string(),
        description: Some("Premium tea from Darjeeling region".to_string()),
        category: "Beverages".to_string(),
        latitude: 27.0360,
        longitude: 88.2627,
        region: "West Bengal".to_string(),
    };

    let product = create_gi_product(&pool, request).await.unwrap();
    assert_eq!(product.name, "Darjeeling Tea");
    assert_eq!(product.category, "Beverages");
    assert_eq!(product.region, "West Bengal");
    assert!(product.metadata_hash.is_some());
    assert!(product.blockchain_tx_id.is_none());

    // Get product by ID
    let retrieved = get_gi_product_by_id(&pool, product.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retrieved.id, product.id);
    assert_eq!(retrieved.name, product.name);
    assert_eq!(retrieved.latitude, 27.0360);
    assert_eq!(retrieved.longitude, 88.2627);

    // Cleanup
    delete_gi_product(&pool, product.id).await.unwrap();
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_create_gi_product_with_invalid_coordinates() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Test coordinates outside India
    let request = CreateGIProductRequest {
        name: "Invalid Product".to_string(),
        description: None,
        category: "Test".to_string(),
        latitude: 50.0, // Outside India
        longitude: 100.0,
        region: "Invalid".to_string(),
    };

    let result = create_gi_product(&pool, request).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("outside valid Indian geographic boundaries"));
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_list_gi_products() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create multiple test products
    let id1 = create_test_product(&pool, "Product 1").await;
    let id2 = create_test_product(&pool, "Product 2").await;
    let id3 = create_test_product(&pool, "Product 3").await;

    // List products
    let products = list_gi_products(&pool, 10, 0).await.unwrap();
    assert!(products.len() >= 3);

    // Test pagination
    let page1 = list_gi_products(&pool, 2, 0).await.unwrap();
    assert_eq!(page1.len(), 2);

    let page2 = list_gi_products(&pool, 2, 2).await.unwrap();
    assert!(page2.len() >= 1);

    // Cleanup
    delete_gi_product(&pool, id1).await.unwrap();
    delete_gi_product(&pool, id2).await.unwrap();
    delete_gi_product(&pool, id3).await.unwrap();
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_update_gi_product() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create test product
    let product_id = create_test_product(&pool, "Original Name").await;
    let original = get_gi_product_by_id(&pool, product_id)
        .await
        .unwrap()
        .unwrap();

    // Update name and description
    let update_request = UpdateGIProductRequest {
        name: Some("Updated Name".to_string()),
        description: Some("New description".to_string()),
        category: None,
        latitude: None,
        longitude: None,
        region: None,
    };
    let updated = update_gi_product(&pool, product_id, update_request)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(updated.name, "Updated Name");
    assert_eq!(updated.description, Some("New description".to_string()));
    assert_eq!(updated.category, original.category);
    assert_ne!(updated.metadata_hash, original.metadata_hash);
    assert!(updated.blockchain_tx_id.is_none()); // Reset after update

    // Cleanup
    delete_gi_product(&pool, product_id).await.unwrap();
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_update_gi_product_with_invalid_coordinates() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create test product
    let product_id = create_test_product(&pool, "Test Product").await;

    // Try to update with invalid coordinates
    let update_request = UpdateGIProductRequest {
        name: None,
        description: None,
        category: None,
        latitude: Some(50.0), // Outside India
        longitude: Some(100.0),
        region: None,
    };
    let result = update_gi_product(&pool, product_id, update_request).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("outside valid Indian geographic boundaries"));

    // Cleanup
    delete_gi_product(&pool, product_id).await.unwrap();
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_delete_gi_product() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create test product
    let product_id = create_test_product(&pool, "To Delete").await;

    // Verify it exists
    let product = get_gi_product_by_id(&pool, product_id).await.unwrap();
    assert!(product.is_some());

    // Delete it
    let deleted = delete_gi_product(&pool, product_id).await.unwrap();
    assert!(deleted);

    // Verify it's gone
    let product = get_gi_product_by_id(&pool, product_id).await.unwrap();
    assert!(product.is_none());

    // Try to delete again
    let deleted = delete_gi_product(&pool, product_id).await.unwrap();
    assert!(!deleted);
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_query_gi_products_by_region() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create products in different regions
    let request1 = CreateGIProductRequest {
        name: "Kanchipuram Silk".to_string(),
        description: Some("Traditional silk from Kanchipuram".to_string()),
        category: "Textiles".to_string(),
        latitude: 12.8342,
        longitude: 79.7036,
        region: "Tamil Nadu".to_string(),
    };
    let product1 = create_gi_product(&pool, request1).await.unwrap();

    let request2 = CreateGIProductRequest {
        name: "Mysore Silk".to_string(),
        description: Some("Silk from Mysore".to_string()),
        category: "Textiles".to_string(),
        latitude: 12.2958,
        longitude: 76.6394,
        region: "Karnataka".to_string(),
    };
    let product2 = create_gi_product(&pool, request2).await.unwrap();

    // Query Tamil Nadu region
    let bounds = GeographicBounds {
        min_lat: 12.0,
        max_lat: 13.0,
        min_lon: 79.0,
        max_lon: 80.0,
    };
    let products = query_gi_products_by_region(&pool, bounds).await.unwrap();
    
    // Should contain product1 but not product2
    assert!(products.iter().any(|p| p.id == product1.id));
    assert!(!products.iter().any(|p| p.id == product2.id));

    // Query Karnataka region
    let bounds = GeographicBounds {
        min_lat: 12.0,
        max_lat: 13.0,
        min_lon: 76.0,
        max_lon: 77.0,
    };
    let products = query_gi_products_by_region(&pool, bounds).await.unwrap();
    
    // Should contain product2 but not product1
    assert!(!products.iter().any(|p| p.id == product1.id));
    assert!(products.iter().any(|p| p.id == product2.id));

    // Cleanup
    delete_gi_product(&pool, product1.id).await.unwrap();
    delete_gi_product(&pool, product2.id).await.unwrap();
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_query_gi_products_by_region_with_invalid_bounds() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Test bounds outside India
    let bounds = GeographicBounds {
        min_lat: 50.0,
        max_lat: 60.0,
        min_lon: 100.0,
        max_lon: 110.0,
    };
    let result = query_gi_products_by_region(&pool, bounds).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("outside valid Indian boundaries"));
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_metadata_hash_computation() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&database_url).await.unwrap();

    // Create two identical products
    let request1 = CreateGIProductRequest {
        name: "Test Product".to_string(),
        description: Some("Description".to_string()),
        category: "Category".to_string(),
        latitude: 28.6139,
        longitude: 77.2090,
        region: "Delhi".to_string(),
    };
    let product1 = create_gi_product(&pool, request1).await.unwrap();

    let request2 = CreateGIProductRequest {
        name: "Test Product".to_string(),
        description: Some("Description".to_string()),
        category: "Category".to_string(),
        latitude: 28.6139,
        longitude: 77.2090,
        region: "Delhi".to_string(),
    };
    let product2 = create_gi_product(&pool, request2).await.unwrap();

    // Hashes should be identical for identical data
    assert_eq!(product1.metadata_hash, product2.metadata_hash);

    // Cleanup
    delete_gi_product(&pool, product1.id).await.unwrap();
    delete_gi_product(&pool, product2.id).await.unwrap();
}
