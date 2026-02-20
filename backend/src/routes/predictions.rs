// ROI Prediction Routes
// Task 12.10: Rust Backend Integration
// Requirements: 3.6, 9.2

use axum::{
    extract::{Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use uuid::Uuid;

use crate::db::Database;
use crate::error::AppError;
use crate::models::prediction::{
    ListPredictionsQuery, ListPredictionsResponse, PredictROIRequest,
    PredictionExplanationResponse, ROIPredictionResponse, ComprehensiveSHAPExplanation,
};

/// Prediction routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/roi", post(predict_roi))
        .route("/:id", get(get_prediction))
        .route("/:id/explanation", get(get_explanation))
        .route("/", get(list_predictions))
}

/// POST /api/v1/predictions/roi - Predict ROI for investment
///
/// Creates a new ROI prediction using the ensemble model (Random Forest, XGBoost, Neural Network).
/// Stores the prediction in the database and submits metadata hash to blockchain.
///
/// # Requirements
/// - Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
async fn predict_roi(
    State(db): State<Database>,
    Json(request): Json<PredictROIRequest>,
) -> Result<Json<ROIPredictionResponse>, AppError> {
    tracing::info!(
        "Received ROI prediction request: sector={}, amount={}",
        request.sector,
        request.investment_amount
    );

    // Validate coordinates
    if !crate::models::validate_indian_coordinates(request.latitude, request.longitude) {
        return Err(AppError::ValidationError(
            "Coordinates must be within Indian geographic boundaries".to_string(),
        ));
    }

    // Validate investment amount
    if request.investment_amount <= 0.0 {
        return Err(AppError::ValidationError(
            "Investment amount must be positive".to_string(),
        ));
    }

    // Validate timeframe
    if request.timeframe_years <= 0 {
        return Err(AppError::ValidationError(
            "Timeframe must be positive".to_string(),
        ));
    }

    // TODO: Call Julia Analytics Engine via gRPC
    // For now, use mock prediction
    tracing::warn!("Using mock prediction - implement gRPC call to Julia Analytics Engine");

    // Mock prediction response (would come from Julia Analytics Engine)
    let predicted_roi = 0.18 + (rand::random::<f64>() * 0.1); // 18-28% ROI
    let confidence_lower = predicted_roi - 0.05;
    let confidence_upper = predicted_roi + 0.05;
    let variance = 0.002;
    let model_version = "1.0.0".to_string();

    // Mock individual predictions
    let individual_predictions = json!({
        "random_forest": predicted_roi - 0.01,
        "xgboost": predicted_roi + 0.01,
        "neural_network": predicted_roi
    });

    // Mock SHAP values
    let shap_values = json!([
        {
            "feature_name": "investment_amount",
            "shap_value": 0.05,
            "feature_value": request.investment_amount
        },
        {
            "feature_name": "latitude",
            "shap_value": 0.03,
            "feature_value": request.latitude
        },
        {
            "feature_name": "sector",
            "shap_value": 0.02,
            "feature_value": request.sector.clone()
        },
        {
            "feature_name": "timeframe_years",
            "shap_value": -0.01,
            "feature_value": request.timeframe_years
        }
    ]);

    // Combine features for storage
    let mut features = json!({
        "latitude": request.latitude,
        "longitude": request.longitude,
        "sector": request.sector,
        "investment_amount": request.investment_amount,
        "timeframe_years": request.timeframe_years
    });

    if let serde_json::Value::Object(ref mut map) = features {
        map.insert(
            "individual_predictions".to_string(),
            individual_predictions.clone(),
        );
        for (key, value) in &request.additional_features {
            map.insert(key.clone(), json!(value));
        }
    }

    // Store prediction in database
    let prediction = crate::db::prediction::create_prediction(
        db.pool(),
        request.latitude,
        request.longitude,
        &request.sector,
        request.investment_amount,
        request.timeframe_years,
        predicted_roi,
        confidence_lower,
        confidence_upper,
        variance,
        &model_version,
        features,
        shap_values,
    )
    .await?;

    // TODO: Submit metadata hash to blockchain (mock for now)
    let blockchain_tx_id = format!("0x{}", uuid::Uuid::new_v4().to_string().replace("-", ""));
    crate::db::prediction::update_blockchain_tx(db.pool(), prediction.id, &blockchain_tx_id)
        .await?;

    tracing::info!(
        "Created ROI prediction: id={}, roi={:.2}%",
        prediction.id,
        predicted_roi * 100.0
    );

    // Fetch updated prediction with blockchain tx
    let updated_prediction = crate::db::prediction::get_prediction_by_id(db.pool(), prediction.id)
        .await?;

    Ok(Json(updated_prediction.to_response()))
}

/// GET /api/v1/predictions/:id - Get prediction by ID
///
/// Retrieves a specific ROI prediction by its ID.
///
/// # Requirements
/// - Requirements 3.6, 9.2
async fn get_prediction(
    State(db): State<Database>,
    Path(id): Path<Uuid>,
) -> Result<Json<ROIPredictionResponse>, AppError> {
    tracing::info!("Fetching prediction: {}", id);

    let prediction = crate::db::prediction::get_prediction_by_id(db.pool(), id).await?;

    Ok(Json(prediction.to_response()))
}

/// GET /api/v1/predictions/:id/explanation - Get SHAP explanation for prediction
///
/// Retrieves the SHAP explanation for a specific ROI prediction, including
/// feature importance, visualizations, and natural language summary.
///
/// # Requirements
/// - Requirements 3.3, 7.1, 7.2, 7.3, 7.8
async fn get_explanation(
    State(db): State<Database>,
    Path(id): Path<Uuid>,
) -> Result<Json<PredictionExplanationResponse>, AppError> {
    tracing::info!("Fetching explanation for prediction: {}", id);

    let prediction = crate::db::prediction::get_prediction_by_id(db.pool(), id).await?;

    // Parse SHAP values from database
    let shap_values = prediction.shap_values.clone();

    // Build comprehensive SHAP explanation
    // In production, this would be retrieved from the Julia Analytics Engine
    let explanation = ComprehensiveSHAPExplanation {
        explanation: crate::models::prediction::SHAPExplanation {
            explanation_type: "local".to_string(),
            prediction: prediction.predicted_roi,
            base_value: 0.15, // Mock base value
            top_features: if let Some(arr) = shap_values.as_array() {
                arr.iter()
                    .take(10)
                    .filter_map(|v| {
                        let obj = v.as_object()?;
                        Some(crate::models::prediction::FeatureImportance {
                            feature_name: obj.get("feature_name")?.as_str()?.to_string(),
                            shap_value: obj.get("shap_value")?.as_f64()?,
                            feature_value: obj.get("feature_value").and_then(|v| v.as_f64()),
                        })
                    })
                    .collect()
            } else {
                vec![]
            },
            summary: format!(
                "This prediction is higher than the base value. The top contributing factors are: \
                 investment amount, location, and sector."
            ),
        },
        visualizations: crate::models::prediction::SHAPVisualizations {
            force_plot: crate::models::prediction::ForcePlotData {
                base_value: 0.15,
                prediction: prediction.predicted_roi,
                features: vec![],
                link: "identity".to_string(),
            },
            waterfall_plot: crate::models::prediction::WaterfallPlotData {
                base_value: 0.15,
                prediction: prediction.predicted_roi,
                features: vec![],
                cumulative_values: vec![],
            },
        },
    };

    Ok(Json(PredictionExplanationResponse {
        prediction_id: prediction.id,
        predicted_roi: prediction.predicted_roi,
        shap_explanation: explanation,
        model_version: prediction.model_version,
    }))
}

/// GET /api/v1/predictions - List predictions with optional filters
///
/// Lists ROI predictions with pagination and optional filtering by sector and ROI range.
async fn list_predictions(
    State(db): State<Database>,
    Query(query): Query<ListPredictionsQuery>,
) -> Result<Json<ListPredictionsResponse>, AppError> {
    tracing::info!(
        "Listing predictions: limit={}, offset={}, sector={:?}",
        query.limit,
        query.offset,
        query.sector
    );

    let (predictions, total) = crate::db::prediction::list_predictions(db.pool(), &query).await?;

    Ok(Json(ListPredictionsResponse {
        predictions,
        total,
        limit: query.limit,
        offset: query.offset,
    }))
}
