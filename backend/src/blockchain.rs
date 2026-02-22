use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Blockchain client for interacting with Substrate node
pub struct BlockchainClient {
    node_url: String,
    retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    GIProduct,
    Cluster,
    ROIPrediction,
    MLAScore,
    Anomaly,
    Forecast,
    RiskAssessment,
    ModelPerformance,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HashSubmission {
    pub hash: String,
    pub data_type: DataType,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HashVerification {
    pub hash: String,
    pub exists: bool,
    pub timestamp: Option<u64>,
}

impl BlockchainClient {
    pub fn new(node_url: String) -> Self {
        Self {
            node_url,
            retry_attempts: 3,
        }
    }

    /// Compute SHA-256 hash of data
    pub fn compute_hash(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// Submit hash to blockchain with retry logic
    pub async fn submit_hash(
        &self,
        data: &[u8],
        data_type: DataType,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let hash = Self::compute_hash(data);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        let submission = HashSubmission {
            hash: hash.clone(),
            data_type,
            timestamp,
        };

        // Retry logic
        for attempt in 1..=self.retry_attempts {
            match self.try_submit(&submission).await {
                Ok(tx_id) => {
                    tracing::info!(
                        "Hash submitted successfully on attempt {}: {}",
                        attempt,
                        tx_id
                    );
                    return Ok(tx_id);
                }
                Err(e) if attempt < self.retry_attempts => {
                    tracing::warn!(
                        "Hash submission attempt {} failed: {}. Retrying...",
                        attempt,
                        e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
                Err(e) => {
                    tracing::error!("Hash submission failed after {} attempts: {}", attempt, e);
                    return Err(e);
                }
            }
        }

        Err("Failed to submit hash after all retry attempts".into())
    }

    /// Internal method to try submitting hash
    async fn try_submit(
        &self,
        submission: &HashSubmission,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // In production, this would use substrate-api-client or similar
        // For now, simulate blockchain submission
        
        tracing::info!(
            "Submitting hash to blockchain: {} (type: {:?})",
            submission.hash,
            submission.data_type
        );

        // Simulate network call
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Generate mock transaction ID
        let tx_id = format!("0x{}", &submission.hash[..16]);
        
        Ok(tx_id)
    }

    /// Verify if hash exists on blockchain
    pub async fn verify_hash(
        &self,
        hash: &str,
    ) -> Result<HashVerification, Box<dyn std::error::Error>> {
        tracing::info!("Verifying hash on blockchain: {}", hash);

        // In production, query the blockchain
        // For now, simulate verification
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        Ok(HashVerification {
            hash: hash.to_string(),
            exists: true, // Simulate that hash exists
            timestamp: Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)?
                    .as_secs(),
            ),
        })
    }

    /// Get blockchain node status
    pub async fn get_node_status(&self) -> Result<NodeStatus, Box<dyn std::error::Error>> {
        tracing::info!("Checking blockchain node status");

        // In production, query actual node
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        Ok(NodeStatus {
            connected: true,
            block_height: 12345,
            node_version: "1.0.0".to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeStatus {
    pub connected: bool,
    pub block_height: u64,
    pub node_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash() {
        let data = b"test data";
        let hash = BlockchainClient::compute_hash(data);
        assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[tokio::test]
    async fn test_submit_hash() {
        let client = BlockchainClient::new("http://localhost:9933".to_string());
        let data = b"test prediction data";
        let result = client.submit_hash(data, DataType::ROIPrediction).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_verify_hash() {
        let client = BlockchainClient::new("http://localhost:9933".to_string());
        let hash = "test_hash_123";
        let result = client.verify_hash(hash).await;
        assert!(result.is_ok());
    }
}
