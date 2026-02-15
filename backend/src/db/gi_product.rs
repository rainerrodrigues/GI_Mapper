use crate::models::{
    CreateGIProductRequest, GIProduct, GeographicBounds, UpdateGIProductRequest,
    validate_indian_coordinates,
};
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use uuid::Uuid;

/// Create a new GI Product in the database
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `request` - GI Product creation request
///
/// # Returns
/// * `Result<GIProduct>` - Created GI Product with generated ID
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 1.2: Validate coordinates fall within valid Indian geographic boundaries
pub async fn create_gi_product(
    pool: &PgPool,
    request: CreateGIProductRequest,
) -> Result<GIProduct> {
    // Validate coordinates are within Indian boundaries (Requirement 1.2)
    if !validate_indian_coordinates(request.latitude, request.longitude) {
        return Err(anyhow!(
            "Coordinates ({}, {}) are outside valid Indian geographic boundaries",
            request.latitude,
            request.longitude
        ));
    }

    // Compute metadata hash for blockchain verification (Requirement 1.4)
    let metadata = format!(
        "{}|{}|{}|{}|{}|{}",
        request.name,
        request.description.as_deref().unwrap_or(""),
        request.category,
        request.latitude,
        request.longitude,
        request.region
    );
    let mut hasher = Sha256::new();
    hasher.update(metadata.as_bytes());
    let metadata_hash = format!("{:x}", hasher.finalize());

    // Insert into database with PostGIS geometry
    let product = sqlx::query_as::<_, GIProduct>(
        r#"
        INSERT INTO gi_products (name, description, category, location, region, metadata_hash)
        VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($4, $5), 4326), $6, $7)
        RETURNING 
            id, 
            name, 
            description, 
            category, 
            ST_Y(location) as latitude,
            ST_X(location) as longitude,
            region, 
            created_at, 
            updated_at, 
            metadata_hash, 
            blockchain_tx_id
        "#,
    )
    .bind(&request.name)
    .bind(&request.description)
    .bind(&request.category)
    .bind(request.longitude) // PostGIS uses (lon, lat) order
    .bind(request.latitude)
    .bind(&request.region)
    .bind(&metadata_hash)
    .fetch_one(pool)
    .await
    .context("Failed to insert GI product into database")?;

    Ok(product)
}

/// Get a GI Product by ID
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `id` - Product UUID
///
/// # Returns
/// * `Result<Option<GIProduct>>` - Product if found, None otherwise
pub async fn get_gi_product_by_id(pool: &PgPool, id: Uuid) -> Result<Option<GIProduct>> {
    let product = sqlx::query_as::<_, GIProduct>(
        r#"
        SELECT 
            id, 
            name, 
            description, 
            category, 
            ST_Y(location) as latitude,
            ST_X(location) as longitude,
            region, 
            created_at, 
            updated_at, 
            metadata_hash, 
            blockchain_tx_id
        FROM gi_products
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .context("Failed to fetch GI product by ID")?;

    Ok(product)
}

/// List all GI Products with optional pagination
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `limit` - Maximum number of products to return
/// * `offset` - Number of products to skip
///
/// # Returns
/// * `Result<Vec<GIProduct>>` - List of GI Products
pub async fn list_gi_products(pool: &PgPool, limit: i64, offset: i64) -> Result<Vec<GIProduct>> {
    let products = sqlx::query_as::<_, GIProduct>(
        r#"
        SELECT 
            id, 
            name, 
            description, 
            category, 
            ST_Y(location) as latitude,
            ST_X(location) as longitude,
            region, 
            created_at, 
            updated_at, 
            metadata_hash, 
            blockchain_tx_id
        FROM gi_products
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        "#,
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await
    .context("Failed to list GI products")?;

    Ok(products)
}

/// Update an existing GI Product
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `id` - Product UUID
/// * `request` - Update request with optional fields
///
/// # Returns
/// * `Result<Option<GIProduct>>` - Updated product if found, None otherwise
///
/// # Requirements
/// * 1.2: Validate coordinates fall within valid Indian geographic boundaries
/// * 1.4: Compute hash of metadata and store on blockchain
pub async fn update_gi_product(
    pool: &PgPool,
    id: Uuid,
    request: UpdateGIProductRequest,
) -> Result<Option<GIProduct>> {
    // Validate coordinates if provided (Requirement 1.2)
    if let (Some(lat), Some(lon)) = (request.latitude, request.longitude) {
        if !validate_indian_coordinates(lat, lon) {
            return Err(anyhow!(
                "Coordinates ({}, {}) are outside valid Indian geographic boundaries",
                lat,
                lon
            ));
        }
    }

    // Fetch existing product to compute new hash
    let existing = get_gi_product_by_id(pool, id).await?;
    if existing.is_none() {
        return Ok(None);
    }
    let existing = existing.unwrap();

    // Merge updates with existing values
    let name = request.name.unwrap_or(existing.name);
    let description = request.description.or(existing.description);
    let category = request.category.unwrap_or(existing.category);
    let latitude = request.latitude.unwrap_or(existing.latitude);
    let longitude = request.longitude.unwrap_or(existing.longitude);
    let region = request.region.unwrap_or(existing.region);

    // Compute new metadata hash (Requirement 1.4)
    let metadata = format!(
        "{}|{}|{}|{}|{}|{}",
        name,
        description.as_deref().unwrap_or(""),
        category,
        latitude,
        longitude,
        region
    );
    let mut hasher = Sha256::new();
    hasher.update(metadata.as_bytes());
    let metadata_hash = format!("{:x}", hasher.finalize());

    // Update in database
    let product = sqlx::query_as::<_, GIProduct>(
        r#"
        UPDATE gi_products
        SET 
            name = $2,
            description = $3,
            category = $4,
            location = ST_SetSRID(ST_MakePoint($5, $6), 4326),
            region = $7,
            updated_at = $8,
            metadata_hash = $9,
            blockchain_tx_id = NULL
        WHERE id = $1
        RETURNING 
            id, 
            name, 
            description, 
            category, 
            ST_Y(location) as latitude,
            ST_X(location) as longitude,
            region, 
            created_at, 
            updated_at, 
            metadata_hash, 
            blockchain_tx_id
        "#,
    )
    .bind(id)
    .bind(&name)
    .bind(&description)
    .bind(&category)
    .bind(longitude) // PostGIS uses (lon, lat) order
    .bind(latitude)
    .bind(&region)
    .bind(Utc::now())
    .bind(&metadata_hash)
    .fetch_optional(pool)
    .await
    .context("Failed to update GI product")?;

    Ok(product)
}

/// Delete a GI Product by ID
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `id` - Product UUID
///
/// # Returns
/// * `Result<bool>` - True if product was deleted, false if not found
pub async fn delete_gi_product(pool: &PgPool, id: Uuid) -> Result<bool> {
    let result = sqlx::query("DELETE FROM gi_products WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .context("Failed to delete GI product")?;

    Ok(result.rows_affected() > 0)
}

/// Query GI Products by geographic region
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `bounds` - Geographic bounding box
///
/// # Returns
/// * `Result<Vec<GIProduct>>` - Products within the specified region
///
/// # Requirements
/// * 1.3: Return all products within specified geographic area
pub async fn query_gi_products_by_region(
    pool: &PgPool,
    bounds: GeographicBounds,
) -> Result<Vec<GIProduct>> {
    // Validate bounds
    if !bounds.is_valid_indian_bounds() {
        return Err(anyhow!(
            "Geographic bounds are outside valid Indian boundaries"
        ));
    }

    // Query using PostGIS spatial operators (Requirement 1.3)
    let products = sqlx::query_as::<_, GIProduct>(
        r#"
        SELECT 
            id, 
            name, 
            description, 
            category, 
            ST_Y(location) as latitude,
            ST_X(location) as longitude,
            region, 
            created_at, 
            updated_at, 
            metadata_hash, 
            blockchain_tx_id
        FROM gi_products
        WHERE ST_Contains(
            ST_MakeEnvelope($1, $2, $3, $4, 4326),
            location
        )
        ORDER BY created_at DESC
        "#,
    )
    .bind(bounds.min_lon)
    .bind(bounds.min_lat)
    .bind(bounds.max_lon)
    .bind(bounds.max_lat)
    .fetch_all(pool)
    .await
    .context("Failed to query GI products by region")?;

    Ok(products)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires running database
    async fn test_create_gi_product() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = PgPool::connect(&database_url).await.unwrap();

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
        assert!(product.metadata_hash.is_some());

        // Cleanup
        delete_gi_product(&pool, product.id).await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires running database
    async fn test_create_gi_product_invalid_coordinates() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = PgPool::connect(&database_url).await.unwrap();

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
    async fn test_query_gi_products_by_region() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = PgPool::connect(&database_url).await.unwrap();

        // Create test product
        let request = CreateGIProductRequest {
            name: "Kanchipuram Silk".to_string(),
            description: Some("Traditional silk from Kanchipuram".to_string()),
            category: "Textiles".to_string(),
            latitude: 12.8342,
            longitude: 79.7036,
            region: "Tamil Nadu".to_string(),
        };
        let product = create_gi_product(&pool, request).await.unwrap();

        // Query by region
        let bounds = GeographicBounds {
            min_lat: 12.0,
            max_lat: 13.0,
            min_lon: 79.0,
            max_lon: 80.0,
        };
        let products = query_gi_products_by_region(&pool, bounds).await.unwrap();
        assert!(products.iter().any(|p| p.id == product.id));

        // Cleanup
        delete_gi_product(&pool, product.id).await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires running database
    async fn test_update_gi_product() {
        dotenv::dotenv().ok();
        let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = PgPool::connect(&database_url).await.unwrap();

        // Create test product
        let request = CreateGIProductRequest {
            name: "Test Product".to_string(),
            description: None,
            category: "Test".to_string(),
            latitude: 28.6139,
            longitude: 77.2090,
            region: "Delhi".to_string(),
        };
        let product = create_gi_product(&pool, request).await.unwrap();

        // Update product
        let update_request = UpdateGIProductRequest {
            name: Some("Updated Product".to_string()),
            description: Some("New description".to_string()),
            category: None,
            latitude: None,
            longitude: None,
            region: None,
        };
        let updated = update_gi_product(&pool, product.id, update_request)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(updated.name, "Updated Product");
        assert_eq!(updated.description, Some("New description".to_string()));
        assert_ne!(updated.metadata_hash, product.metadata_hash);

        // Cleanup
        delete_gi_product(&pool, product.id).await.unwrap();
    }
}
