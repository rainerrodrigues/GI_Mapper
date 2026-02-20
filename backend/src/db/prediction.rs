// Database operations for ROI predictions
// Task 12.10: Rust Backend Integration
// Requirements: 3.6, 9.2

use sqlx::PgPool;
use uuid::Uuid;
use sha2::{Sha256, Digest};
use crate::models::prediction::{
    ROIPrediction, ROIPredictionSummary, ListPredictionsQuery,
};
use crate::error::AppError;

/// Create a new ROI prediction in the database
pub async fn create_prediction(
    pool: &PgPool,
    latitude: f64,
    longitude: f64,
    sector: &str,
    investment_amount: f64,
    timeframe_years: i32,
    predicted_roi: f64,
    confidence_lower: f64,
    confidence_upper: f64,
    variance: f64,
    model_version: &str,
    features: serde_json::Value,
    shap_values: serde_json::Value,
) -> Result<ROIPrediction, AppError> {
    let id = Uuid::new_v4();
    
    // Compute metadata hash for blockchain storage
    let metadata = format!(
        "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
        id, latitude, longitude, sector, investment_amount,
        timeframe_years, predicted_roi, confidence_lower, confidence_upper, model_version
    );
    let mut hasher = Sha256::new();
    hasher.update(metadata.as_bytes());
    let metadata_hash = format!("{:x}", hasher.finalize());
    
    // Create location geometry (Point)
    let location_wkt = format!("POINT({} {})", longitude, latitude);
    
    let prediction = sqlx::query_as::<_, ROIPrediction>(
        r#"
        INSERT INTO roi_predictions (
            id, location, sector, investment_amount, timeframe_years,
            predicted_roi, confidence_lower, confidence_upper, variance,
            model_version, features, shap_values, metadata_hash
        )
        VALUES (
            $1, ST_GeomFromText($2, 4326), $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12, $13
        )
        RETURNING 
            id, 
            ST_X(location::geometry) as longitude,
            ST_Y(location::geometry) as latitude,
            sector, investment_amount, timeframe_years,
            predicted_roi, confidence_lower, confidence_upper, variance,
            model_version, features, shap_values, created_at,
            actual_roi, metadata_hash, blockchain_tx_id
        "#
    )
    .bind(id)
    .bind(&location_wkt)
    .bind(sector)
    .bind(investment_amount)
    .bind(timeframe_years)
    .bind(predicted_roi)
    .bind(confidence_lower)
    .bind(confidence_upper)
    .bind(variance)
    .bind(model_version)
    .bind(&features)
    .bind(&shap_values)
    .bind(&metadata_hash)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create prediction: {}", e);
        AppError::DatabaseError(e.to_string())
    })?;
    
    tracing::info!("Created ROI prediction: {}", id);
    
    Ok(prediction)
}

/// Get a prediction by ID
pub async fn get_prediction_by_id(
    pool: &PgPool,
    id: Uuid,
) -> Result<ROIPrediction, AppError> {
    let prediction = sqlx::query_as::<_, ROIPrediction>(
        r#"
        SELECT 
            id,
            ST_X(location::geometry) as longitude,
            ST_Y(location::geometry) as latitude,
            sector, investment_amount, timeframe_years,
            predicted_roi, confidence_lower, confidence_upper, variance,
            model_version, features, shap_values, created_at,
            actual_roi, metadata_hash, blockchain_tx_id
        FROM roi_predictions
        WHERE id = $1
        "#
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to fetch prediction: {}", e);
        AppError::DatabaseError(e.to_string())
    })?
    .ok_or_else(|| AppError::NotFound(format!("Prediction {} not found", id)))?;
    
    Ok(prediction)
}

/// List predictions with optional filters
pub async fn list_predictions(
    pool: &PgPool,
    query: &ListPredictionsQuery,
) -> Result<(Vec<ROIPredictionSummary>, i64), AppError> {
    // Build WHERE clause based on filters
    let mut where_clauses = vec![];
    let mut param_count = 1;
    
    if query.sector.is_some() {
        where_clauses.push(format!("sector = ${}", param_count));
        param_count += 1;
    }
    
    if query.min_roi.is_some() {
        where_clauses.push(format!("predicted_roi >= ${}", param_count));
        param_count += 1;
    }
    
    if query.max_roi.is_some() {
        where_clauses.push(format!("predicted_roi <= ${}", param_count));
        param_count += 1;
    }
    
    let where_clause = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };
    
    // Get total count
    let count_query = format!(
        "SELECT COUNT(*) as count FROM roi_predictions {}",
        where_clause
    );
    
    let mut count_query_builder = sqlx::query_scalar::<_, i64>(&count_query);
    
    if let Some(ref sector) = query.sector {
        count_query_builder = count_query_builder.bind(sector);
    }
    if let Some(min_roi) = query.min_roi {
        count_query_builder = count_query_builder.bind(min_roi);
    }
    if let Some(max_roi) = query.max_roi {
        count_query_builder = count_query_builder.bind(max_roi);
    }
    
    let total = count_query_builder
        .fetch_one(pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to count predictions: {}", e);
            AppError::DatabaseError(e.to_string())
        })?;
    
    // Get predictions
    let list_query = format!(
        r#"
        SELECT 
            id,
            ST_X(location::geometry) as longitude,
            ST_Y(location::geometry) as latitude,
            sector, investment_amount,
            predicted_roi, confidence_lower, confidence_upper,
            model_version, created_at
        FROM roi_predictions
        {}
        ORDER BY created_at DESC
        LIMIT ${} OFFSET ${}
        "#,
        where_clause,
        param_count,
        param_count + 1
    );
    
    let mut list_query_builder = sqlx::query_as::<_, ROIPredictionSummary>(&list_query);
    
    if let Some(ref sector) = query.sector {
        list_query_builder = list_query_builder.bind(sector);
    }
    if let Some(min_roi) = query.min_roi {
        list_query_builder = list_query_builder.bind(min_roi);
    }
    if let Some(max_roi) = query.max_roi {
        list_query_builder = list_query_builder.bind(max_roi);
    }
    
    list_query_builder = list_query_builder
        .bind(query.limit)
        .bind(query.offset);
    
    let predictions = list_query_builder
        .fetch_all(pool)
        .await
        .map_err(|e| {
            tracing::error!("Failed to list predictions: {}", e);
            AppError::DatabaseError(e.to_string())
        })?;
    
    Ok((predictions, total))
}

/// Update blockchain transaction ID for a prediction
pub async fn update_blockchain_tx(
    pool: &PgPool,
    id: Uuid,
    blockchain_tx_id: &str,
) -> Result<(), AppError> {
    sqlx::query(
        "UPDATE roi_predictions SET blockchain_tx_id = $1 WHERE id = $2"
    )
    .bind(blockchain_tx_id)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update blockchain tx: {}", e);
        AppError::DatabaseError(e.to_string())
    })?;
    
    Ok(())
}

/// Update actual ROI for a prediction (for model performance tracking)
pub async fn update_actual_roi(
    pool: &PgPool,
    id: Uuid,
    actual_roi: f64,
) -> Result<(), AppError> {
    sqlx::query(
        "UPDATE roi_predictions SET actual_roi = $1 WHERE id = $2"
    )
    .bind(actual_roi)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update actual ROI: {}", e);
        AppError::DatabaseError(e.to_string())
    })?;
    
    tracing::info!("Updated actual ROI for prediction {}: {}", id, actual_roi);
    
    Ok(())
}
