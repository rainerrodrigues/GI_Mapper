# Data Models

This module contains the data models for the AI-Powered Blockchain GIS Platform.

## GI Product Models

### `GIProduct`

Represents a Geographical Indication product with spatial coordinates.

**Fields:**
- `id`: UUID - Unique identifier
- `name`: String - Product name
- `description`: Option<String> - Optional product description
- `category`: String - Product category (e.g., "Beverages", "Textiles")
- `latitude`: f64 - Latitude coordinate (WGS84)
- `longitude`: f64 - Longitude coordinate (WGS84)
- `region`: String - Geographic region (e.g., "West Bengal")
- `created_at`: DateTime<Utc> - Creation timestamp
- `updated_at`: DateTime<Utc> - Last update timestamp
- `metadata_hash`: Option<String> - SHA-256 hash for blockchain verification
- `blockchain_tx_id`: Option<String> - Substrate blockchain transaction ID

### `CreateGIProductRequest`

Request payload for creating a new GI Product.

**Fields:**
- `name`: String - Product name (required)
- `description`: Option<String> - Product description
- `category`: String - Product category (required)
- `latitude`: f64 - Latitude coordinate (required)
- `longitude`: f64 - Longitude coordinate (required)
- `region`: String - Geographic region (required)

**Validation:**
- Coordinates must be within valid Indian geographic boundaries (6°N-38°N, 66°E-99°E)

### `UpdateGIProductRequest`

Request payload for updating an existing GI Product.

**Fields:**
- `name`: Option<String> - New product name
- `description`: Option<String> - New product description
- `category`: Option<String> - New product category
- `latitude`: Option<f64> - New latitude coordinate
- `longitude`: Option<f64> - New longitude coordinate
- `region`: Option<String> - New geographic region

**Validation:**
- If coordinates are provided, they must be within valid Indian geographic boundaries

### `GeographicBounds`

Represents a geographic bounding box for spatial queries.

**Fields:**
- `min_lat`: f64 - Minimum latitude
- `max_lat`: f64 - Maximum latitude
- `min_lon`: f64 - Minimum longitude
- `max_lon`: f64 - Maximum longitude

**Methods:**
- `is_valid_indian_bounds()`: Validates that bounds are within Indian territory

### Helper Functions

#### `validate_indian_coordinates(latitude: f64, longitude: f64) -> bool`

Validates that geographic coordinates fall within valid Indian boundaries.

**Parameters:**
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate

**Returns:**
- `true` if coordinates are within India (approximately 6°N-38°N, 66°E-99°E)
- `false` otherwise

## Requirements Mapping

- **Requirement 1.1**: Store GI product data in Geospatial_Store
- **Requirement 1.2**: Validate coordinates fall within valid Indian geographic boundaries
- **Requirement 1.3**: Return all products within specified geographic area
- **Requirement 1.4**: Compute hash of metadata for blockchain verification
