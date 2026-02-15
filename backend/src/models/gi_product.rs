use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

/// GI Product data model
/// Represents a Geographical Indication product with spatial coordinates
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct GIProduct {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    #[sqlx(skip)]
    pub latitude: f64,
    #[sqlx(skip)]
    pub longitude: f64,
    pub region: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata_hash: Option<String>,
    pub blockchain_tx_id: Option<String>,
}

/// Request payload for creating a new GI Product
#[derive(Debug, Deserialize)]
pub struct CreateGIProductRequest {
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub latitude: f64,
    pub longitude: f64,
    pub region: String,
}

/// Request payload for updating an existing GI Product
#[derive(Debug, Deserialize)]
pub struct UpdateGIProductRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub category: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub region: Option<String>,
}

/// Geographic bounds for spatial queries
#[derive(Debug, Deserialize)]
pub struct GeographicBounds {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl GeographicBounds {
    /// Validate that bounds are within valid Indian geographic boundaries
    /// India approximately: 8°N to 37°N latitude, 68°E to 97°E longitude
    pub fn is_valid_indian_bounds(&self) -> bool {
        self.min_lat >= 6.0
            && self.max_lat <= 38.0
            && self.min_lon >= 66.0
            && self.max_lon <= 99.0
            && self.min_lat < self.max_lat
            && self.min_lon < self.max_lon
    }
}

/// Validate geographic coordinates are within valid Indian boundaries
pub fn validate_indian_coordinates(latitude: f64, longitude: f64) -> bool {
    // India approximately: 8°N to 37°N latitude, 68°E to 97°E longitude
    // Using slightly wider bounds to include territories
    latitude >= 6.0 && latitude <= 38.0 && longitude >= 66.0 && longitude <= 99.0
}
