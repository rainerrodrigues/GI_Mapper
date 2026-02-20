// ROI Prediction Data Models
// Task 12.10: Rust Backend Integration
// Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.9

use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Request to predict ROI
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictROIRequest {
    pub latitude: f64,
    pub longitude: f64,
    pub sector: String,
    pub investment_amount: f64,
    pub timeframe_years: i32,
    #[serde(default)]
    pub additional_features: HashMap<String, f64>,
}

/// Individual model prediction
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndividualPredictions {
    pub random_forest: f64,
    pub xgboost: f64,
    pub neural_network: f64,
}

/// Feature importance from SHAP
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureImportance {
    pub feature_name: String,
    pub shap_value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_value: Option<f64>,
}

/// SHAP explanation details
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SHAPExplanation {
    pub explanation_type: String,
    pub prediction: f64,
    pub base_value: f64,
    pub top_features: Vec<FeatureImportance>,
    pub summary: String,
}

/// Force plot visualization data
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ForcePlotData {
    pub base_value: f64,
    pub prediction: f64,
    pub features: Vec<serde_json::Value>,
    pub link: String,
}

/// Waterfall plot visualization data
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WaterfallPlotData {
    pub base_value: f64,
    pub prediction: f64,
    pub features: Vec<serde_json::Value>,
    pub cumulative_values: Vec<f64>,
}

/// Complete SHAP visualization data
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SHAPVisualizations {
    pub force_plot: ForcePlotData,
    pub waterfall_plot: WaterfallPlotData,
}

/// Complete SHAP explanation with visualizations
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComprehensiveSHAPExplanation {
    pub explanation: SHAPExplanation,
    pub visualizations: SHAPVisualizations,
}

/// ROI prediction response
#[derive(Debug, Serialize, Deserialize)]
pub struct ROIPredictionResponse {
    pub id: Uuid,
    pub predicted_roi: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub variance: f64,
    pub individual_predictions: IndividualPredictions,
    pub model_version: String,
    pub feature_importance: Vec<FeatureImportance>,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blockchain_tx_id: Option<String>,
}

/// ROI prediction stored in database
#[derive(Debug, FromRow)]
pub struct ROIPrediction {
    pub id: Uuid,
    pub latitude: f64,
    pub longitude: f64,
    pub sector: String,
    pub investment_amount: f64,
    pub timeframe_years: i32,
    pub predicted_roi: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub variance: f64,
    pub model_version: String,
    pub features: sqlx::types::JsonValue,
    pub shap_values: sqlx::types::JsonValue,
    pub created_at: DateTime<Utc>,
    pub actual_roi: Option<f64>,
    pub metadata_hash: Option<String>,
    pub blockchain_tx_id: Option<String>,
}

/// Prediction explanation response
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionExplanationResponse {
    pub prediction_id: Uuid,
    pub predicted_roi: f64,
    pub shap_explanation: ComprehensiveSHAPExplanation,
    pub model_version: String,
}

/// List predictions query parameters
#[derive(Debug, Deserialize)]
pub struct ListPredictionsQuery {
    #[serde(default = "default_limit")]
    pub limit: i64,
    #[serde(default)]
    pub offset: i64,
    pub sector: Option<String>,
    pub min_roi: Option<f64>,
    pub max_roi: Option<f64>,
}

fn default_limit() -> i64 {
    50
}

/// List predictions response
#[derive(Debug, Serialize)]
pub struct ListPredictionsResponse {
    pub predictions: Vec<ROIPredictionSummary>,
    pub total: i64,
    pub limit: i64,
    pub offset: i64,
}

/// Summary of ROI prediction for list view
#[derive(Debug, Serialize, FromRow)]
pub struct ROIPredictionSummary {
    pub id: Uuid,
    pub latitude: f64,
    pub longitude: f64,
    pub sector: String,
    pub investment_amount: f64,
    pub predicted_roi: f64,
    pub confidence_lower: f64,
    pub confidence_upper: f64,
    pub model_version: String,
    pub created_at: DateTime<Utc>,
}

impl ROIPrediction {
    /// Convert to response format
    pub fn to_response(&self) -> ROIPredictionResponse {
        // Parse individual predictions from features JSON
        let individual_predictions = if let Some(obj) = self.features.as_object() {
            IndividualPredictions {
                random_forest: obj.get("random_forest")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(self.predicted_roi),
                xgboost: obj.get("xgboost")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(self.predicted_roi),
                neural_network: obj.get("neural_network")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(self.predicted_roi),
            }
        } else {
            IndividualPredictions {
                random_forest: self.predicted_roi,
                xgboost: self.predicted_roi,
                neural_network: self.predicted_roi,
            }
        };

        // Parse feature importance from SHAP values JSON
        let feature_importance = if let Some(arr) = self.shap_values.as_array() {
            arr.iter()
                .filter_map(|v| {
                    let obj = v.as_object()?;
                    Some(FeatureImportance {
                        feature_name: obj.get("feature_name")?.as_str()?.to_string(),
                        shap_value: obj.get("shap_value")?.as_f64()?,
                        feature_value: obj.get("feature_value").and_then(|v| v.as_f64()),
                    })
                })
                .collect()
        } else {
            vec![]
        };

        ROIPredictionResponse {
            id: self.id,
            predicted_roi: self.predicted_roi,
            confidence_lower: self.confidence_lower,
            confidence_upper: self.confidence_upper,
            variance: self.variance,
            individual_predictions,
            model_version: self.model_version.clone(),
            feature_importance,
            created_at: self.created_at,
            blockchain_tx_id: self.blockchain_tx_id.clone(),
        }
    }
}
