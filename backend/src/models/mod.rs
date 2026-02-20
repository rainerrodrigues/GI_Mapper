pub mod gi_product;
pub mod user;
pub mod cluster;
pub mod prediction;

pub use gi_product::{
    CreateGIProductRequest, GIProduct, GeographicBounds, UpdateGIProductRequest,
    validate_indian_coordinates,
};
pub use user::{
    Claims, CreateUserRequest, LoginRequest, LoginResponse, User, UserInfo, UserRole,
};
pub use cluster::{
    Cluster, ClusterInfo, ClusterListResponse, ClusterSummary, DataPoint,
    DetectClustersRequest, DetectClustersResponse, QualityMetrics,
};
pub use prediction::{
    PredictROIRequest, ROIPrediction, ROIPredictionResponse, ROIPredictionSummary,
    PredictionExplanationResponse, ListPredictionsQuery, ListPredictionsResponse,
    FeatureImportance, IndividualPredictions, SHAPExplanation, ComprehensiveSHAPExplanation,
};
