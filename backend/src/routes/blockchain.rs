use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    blockchain::{BlockchainClient, DataType},
    db::Database,
    error::AppError,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitHashRequest {
    pub data: String,
    pub data_type: DataType,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitHashResponse {
    pub hash: String,
    pub transaction_id: String,
    pub success: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VerifyHashResponse {
    pub hash: String,
    pub exists: bool,
    pub timestamp: Option<u64>,
}

pub fn routes() -> Router<Database> {
    Router::new()
        .route("/submit", post(submit_hash))
        .route("/verify/:hash", get(verify_hash))
        .route("/status", get(get_node_status))
}

/// Submit data hash to blockchain
async fn submit_hash(
    State(_database): State<Database>,
    Json(payload): Json<SubmitHashRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Create blockchain client (in production, this would be a shared instance)
    let client = BlockchainClient::new("http://localhost:9933".to_string());
    
    let data_bytes = payload.data.as_bytes();
    let hash = BlockchainClient::compute_hash(data_bytes);

    let tx_id = client
        .submit_hash(data_bytes, payload.data_type)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    Ok((
        StatusCode::OK,
        Json(SubmitHashResponse {
            hash,
            transaction_id: tx_id,
            success: true,
        }),
    ))
}

/// Verify hash exists on blockchain
async fn verify_hash(
    State(_database): State<Database>,
    Path(hash): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let client = BlockchainClient::new("http://localhost:9933".to_string());
    
    let verification = client
        .verify_hash(&hash)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    Ok((
        StatusCode::OK,
        Json(VerifyHashResponse {
            hash: verification.hash,
            exists: verification.exists,
            timestamp: verification.timestamp,
        }),
    ))
}

/// Get blockchain node status
async fn get_node_status(
    State(_database): State<Database>,
) -> Result<impl IntoResponse, AppError> {
    let client = BlockchainClient::new("http://localhost:9933".to_string());
    
    let status = client
        .get_node_status()
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    Ok((StatusCode::OK, Json(status)))
}
