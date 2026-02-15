# Database Operations

This module contains database operations for the AI-Powered Blockchain GIS Platform.

## GI Product Operations

The `gi_product` module provides CRUD operations and spatial queries for Geographical Indication products.

### Functions

#### `create_gi_product(pool: &PgPool, request: CreateGIProductRequest) -> Result<GIProduct>`

Creates a new GI Product in the database.

**Features:**
- Validates coordinates are within Indian geographic boundaries
- Computes SHA-256 hash of metadata for blockchain verification
- Stores location as PostGIS Point geometry (SRID 4326)
- Returns created product with generated UUID

**Requirements:**
- 1.1: Store GI product data in Geospatial_Store
- 1.2: Validate coordinates fall within valid Indian geographic boundaries
- 1.4: Compute hash of metadata for blockchain verification

**Example:**
```rust
let request = CreateGIProductRequest {
    name: "Darjeeling Tea".to_string(),
    description: Some("Premium tea from Darjeeling".to_string()),
    category: "Beverages".to_string(),
    latitude: 27.0360,
    longitude: 88.2627,
    region: "West Bengal".to_string(),
};
let product = create_gi_product(&pool, request).await?;
```

#### `get_gi_product_by_id(pool: &PgPool, id: Uuid) -> Result<Option<GIProduct>>`

Retrieves a GI Product by its UUID.

**Returns:**
- `Some(GIProduct)` if found
- `None` if not found

**Example:**
```rust
let product = get_gi_product_by_id(&pool, product_id).await?;
if let Some(p) = product {
    println!("Found product: {}", p.name);
}
```

#### `list_gi_products(pool: &PgPool, limit: i64, offset: i64) -> Result<Vec<GIProduct>>`

Lists all GI Products with pagination support.

**Parameters:**
- `limit`: Maximum number of products to return
- `offset`: Number of products to skip

**Returns:**
- Vector of GI Products ordered by creation date (newest first)

**Example:**
```rust
// Get first 10 products
let products = list_gi_products(&pool, 10, 0).await?;

// Get next 10 products
let products = list_gi_products(&pool, 10, 10).await?;
```

#### `update_gi_product(pool: &PgPool, id: Uuid, request: UpdateGIProductRequest) -> Result<Option<GIProduct>>`

Updates an existing GI Product.

**Features:**
- Validates new coordinates if provided
- Merges updates with existing values
- Recomputes metadata hash
- Resets blockchain_tx_id (requires re-verification)
- Updates updated_at timestamp

**Requirements:**
- 1.2: Validate coordinates fall within valid Indian geographic boundaries
- 1.4: Compute hash of metadata for blockchain verification

**Returns:**
- `Some(GIProduct)` if product was found and updated
- `None` if product not found

**Example:**
```rust
let update = UpdateGIProductRequest {
    name: Some("Updated Name".to_string()),
    description: Some("New description".to_string()),
    category: None,
    latitude: None,
    longitude: None,
    region: None,
};
let updated = update_gi_product(&pool, product_id, update).await?;
```

#### `delete_gi_product(pool: &PgPool, id: Uuid) -> Result<bool>`

Deletes a GI Product by its UUID.

**Returns:**
- `true` if product was deleted
- `false` if product was not found

**Example:**
```rust
let deleted = delete_gi_product(&pool, product_id).await?;
if deleted {
    println!("Product deleted successfully");
}
```

#### `query_gi_products_by_region(pool: &PgPool, bounds: GeographicBounds) -> Result<Vec<GIProduct>>`

Queries GI Products within a geographic bounding box using PostGIS spatial operators.

**Features:**
- Validates bounds are within Indian territory
- Uses PostGIS ST_Contains for spatial query
- Returns products ordered by creation date

**Requirements:**
- 1.3: Return all products within specified geographic area

**Example:**
```rust
let bounds = GeographicBounds {
    min_lat: 12.0,
    max_lat: 13.0,
    min_lon: 79.0,
    max_lon: 80.0,
};
let products = query_gi_products_by_region(&pool, bounds).await?;
println!("Found {} products in region", products.len());
```

## PostGIS Integration

All spatial operations use PostGIS functions:

- **ST_SetSRID**: Sets the spatial reference system (SRID 4326 for WGS84)
- **ST_MakePoint**: Creates a Point geometry from longitude and latitude
- **ST_X / ST_Y**: Extracts longitude and latitude from Point geometry
- **ST_MakeEnvelope**: Creates a bounding box from coordinates
- **ST_Contains**: Tests if a geometry contains another geometry

## Database Schema

The `gi_products` table structure:

```sql
CREATE TABLE gi_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    location GEOMETRY(Point, 4326) NOT NULL,
    region VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128)
);

-- Spatial index for location-based queries
CREATE INDEX idx_gi_products_location ON gi_products USING GIST(location);
```

## Testing

Unit tests are provided in `backend/tests/gi_product_test.rs`. Tests require a running PostgreSQL database with PostGIS extension.

Run tests with:
```bash
cargo test --test gi_product_test -- --ignored
```

Tests cover:
- Creating products with valid and invalid coordinates
- Retrieving products by ID
- Listing products with pagination
- Updating products
- Deleting products
- Spatial queries by region
- Metadata hash computation
