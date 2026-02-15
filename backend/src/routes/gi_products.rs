use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    middleware,
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    auth_middleware::{require_analyst, require_viewer},
    db::{gi_product, Database},
    error::AppError,
    models::{CreateGIProductRequest, GIProduct, GeographicBounds, UpdateGIProductRequest},
};

/// GI Products routes
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 1.3: Return all products within specified geographic area
/// * 9.2: Provide RESTful endpoints for all platform operations
/// * 14.3: Support role-based access control
/// * 14.4: Verify user has appropriate permissions for operations
pub fn routes() -> Router<Database> {
    Router::new()
        // Read operations - require viewer role or higher
        .route("/", get(list))
        .route("/:id", get(get_by_id))
        .route("/region/:bounds", get(get_by_region))
        .route_layer(middleware::from_fn(require_viewer))
        // Write operations - require analyst role or higher
        .route("/", post(create))
        .route("/:id", put(update).delete(delete_product))
        .route_layer(middleware::from_fn(require_analyst))
}

/// Query parameters for listing GI products
#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(default = "default_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}

fn default_limit() -> i64 {
    100
}

/// Create a new GI product
///
/// POST /api/v1/gi-products
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 1.2: Validate coordinates fall within valid Indian geographic boundaries
/// * 9.2: Provide RESTful endpoints
/// * 9.4: Return responses in JSON format with appropriate HTTP status codes
async fn create(
    State(database): State<Database>,
    Json(request): Json<CreateGIProductRequest>,
) -> Result<(StatusCode, Json<GIProduct>), AppError> {
    let product = gi_product::create_gi_product(database.pool(), request)
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(product)))
}

/// List all GI products with pagination
///
/// GET /api/v1/gi-products?limit=100&offset=0
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 9.2: Provide RESTful endpoints
/// * 9.4: Return responses in JSON format
async fn list(
    State(database): State<Database>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<GIProduct>>, AppError> {
    let products = gi_product::list_gi_products(database.pool(), query.limit, query.offset)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    Ok(Json(products))
}

/// Get a GI product by ID
///
/// GET /api/v1/gi-products/:id
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 9.2: Provide RESTful endpoints
/// * 9.4: Return responses in JSON format with appropriate HTTP status codes
async fn get_by_id(
    State(database): State<Database>,
    Path(id): Path<Uuid>,
) -> Result<Json<GIProduct>, AppError> {
    let product = gi_product::get_gi_product_by_id(database.pool(), id)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?
        .ok_or_else(|| AppError::NotFound(format!("GI product with ID {} not found", id)))?;

    Ok(Json(product))
}

/// Update a GI product by ID
///
/// PUT /api/v1/gi-products/:id
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 1.2: Validate coordinates fall within valid Indian geographic boundaries
/// * 9.2: Provide RESTful endpoints
/// * 9.4: Return responses in JSON format with appropriate HTTP status codes
async fn update(
    State(database): State<Database>,
    Path(id): Path<Uuid>,
    Json(request): Json<UpdateGIProductRequest>,
) -> Result<Json<GIProduct>, AppError> {
    let product = gi_product::update_gi_product(database.pool(), id, request)
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .ok_or_else(|| AppError::NotFound(format!("GI product with ID {} not found", id)))?;

    Ok(Json(product))
}

/// Delete a GI product by ID
///
/// DELETE /api/v1/gi-products/:id
///
/// # Requirements
/// * 1.1: Store GI product data in Geospatial_Store
/// * 9.2: Provide RESTful endpoints
/// * 9.4: Return responses in JSON format with appropriate HTTP status codes
async fn delete_product(
    State(database): State<Database>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let deleted = gi_product::delete_gi_product(database.pool(), id)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "GI product with ID {} not found",
            id
        )))
    }
}

/// Get GI products by geographic region
///
/// GET /api/v1/gi-products/region/:bounds
///
/// Bounds format: "min_lat,min_lon,max_lat,max_lon"
/// Example: /api/v1/gi-products/region/12.0,79.0,13.0,80.0
///
/// # Requirements
/// * 1.3: Return all products within specified geographic area
/// * 9.2: Provide RESTful endpoints
/// * 9.4: Return responses in JSON format with appropriate HTTP status codes
async fn get_by_region(
    State(database): State<Database>,
    Path(bounds_str): Path<String>,
) -> Result<Json<Vec<GIProduct>>, AppError> {
    // Parse bounds string: "min_lat,min_lon,max_lat,max_lon"
    let parts: Vec<&str> = bounds_str.split(',').collect();
    if parts.len() != 4 {
        return Err(AppError::BadRequest(
            "Invalid bounds format. Expected: min_lat,min_lon,max_lat,max_lon".to_string(),
        ));
    }

    let bounds = GeographicBounds {
        min_lat: parts[0]
            .parse()
            .map_err(|_| AppError::BadRequest("Invalid min_lat value".to_string()))?,
        min_lon: parts[1]
            .parse()
            .map_err(|_| AppError::BadRequest("Invalid min_lon value".to_string()))?,
        max_lat: parts[2]
            .parse()
            .map_err(|_| AppError::BadRequest("Invalid max_lat value".to_string()))?,
        max_lon: parts[3]
            .parse()
            .map_err(|_| AppError::BadRequest("Invalid max_lon value".to_string()))?,
    };

    let products = gi_product::query_gi_products_by_region(database.pool(), bounds)
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    Ok(Json(products))
}
