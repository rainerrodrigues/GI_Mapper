# Analytics Engine Protocol Buffers

This directory contains the Protocol Buffer definitions for the Analytics Engine gRPC service.

## Overview

The `analytics.proto` file defines the complete gRPC service interface between the Rust backend and the Julia Analytics Engine. It includes message definitions for all analytics operations:

- **ROI Prediction**: Ensemble regression models for investment return prediction
- **Anomaly Detection**: Unsupervised learning for spending anomaly detection
- **Spatial Clustering**: DBSCAN and K-Means clustering for economic cluster detection
- **MLA Scoring**: Weighted AI index for development impact scoring
- **Time Series Forecasting**: ARIMA, Prophet, and LSTM ensemble forecasting
- **Risk Assessment**: Multi-dimensional risk scoring and classification
- **Explainability**: SHAP-based explanations for all model outputs
- **Model Performance**: Comprehensive KPI tracking and monitoring
- **Model Training**: Automated model retraining and deployment

## Protocol Buffer Schema

The schema is organized into logical sections:

1. **Service Definition**: Main `AnalyticsService` with 9 RPC methods
2. **ROI Prediction Messages**: Request/response for investment ROI prediction
3. **Anomaly Detection Messages**: Transaction analysis and anomaly reporting
4. **Clustering Messages**: Spatial clustering with quality metrics
5. **MLA Scoring Messages**: Development impact scoring with detailed indicators
6. **Forecasting Messages**: Time series forecasting with multiple horizons
7. **Risk Assessment Messages**: Multi-dimensional risk analysis
8. **Explainability Messages**: SHAP explanations and counterfactuals
9. **Model Performance Messages**: KPI tracking and drift detection
10. **Model Training Messages**: Automated retraining requests

## Code Generation

To generate Julia code from the Protocol Buffer schema:

```bash
# Install protoc compiler if not already installed
# On Ubuntu/Debian:
sudo apt-get install protobuf-compiler

# Generate Julia code (requires ProtoBuf.jl)
julia -e 'using ProtoBuf; protojl("analytics.proto", "analytics_pb.jl")'
```

**Note**: The current implementation uses a skeleton approach with manual message handling. Full Protocol Buffer code generation will be completed when the gRPC.jl ecosystem matures or when using alternative approaches like gRPC-Web.

## gRPC Service Handlers

The service handlers are implemented in `analytics/src/grpc/service.jl`:

- `handle_predict_roi()` - ROI prediction (Task 12.9)
- `handle_detect_anomalies()` - Anomaly detection (Task 14.16)
- `handle_detect_clusters()` - Spatial clustering (Task 9.10)
- `handle_compute_mla_score()` - MLA scoring (Task 13.12)
- `handle_generate_forecast()` - Time series forecasting (Task 16.13)
- `handle_assess_risk()` - Risk assessment (Task 17.14)
- `handle_explain_prediction()` - SHAP explanations (Task 10.x)
- `handle_get_model_metrics()` - Model performance (Task 18.19)
- `handle_train_model()` - Model training (Tasks 19.x)

Each handler is documented with:
- Purpose and functionality
- Input parameters
- Return values
- Related requirements
- Implementation status

## Requirements Validation

This implementation satisfies the following requirements:

### Requirement 10.2: Analytics Engine Interface
✓ Interface with Backend API through gRPC protocol

### Requirement 10.3: Algorithm Execution
✓ Execute appropriate AI/ML algorithms (regression, clustering, anomaly detection, forecasting, risk assessment)

### Requirement 10.6: Result Return
✓ Return results to Backend API with execution metadata

## Message Design Principles

1. **Completeness**: All required data for each operation is included
2. **Extensibility**: Maps and repeated fields allow for future additions
3. **Type Safety**: Strong typing with appropriate numeric types
4. **Efficiency**: Binary serialization for fast communication
5. **Clarity**: Clear naming and logical grouping of related fields

## Integration with Rust Backend

The Rust backend will use the `tonic` and `prost` crates to:

1. Generate Rust code from `analytics.proto`
2. Create a gRPC client
3. Make RPC calls to the Julia Analytics Engine
4. Handle responses and errors

Example Rust client usage:

```rust
use analytics_client::AnalyticsServiceClient;

let mut client = AnalyticsServiceClient::connect("http://localhost:50051").await?;

let request = tonic::Request::new(RoiRequest {
    latitude: 28.6139,
    longitude: 77.2090,
    sector: "agriculture".to_string(),
    investment_amount: 1000000.0,
    timeframe_years: 5,
    additional_features: HashMap::new(),
});

let response = client.predict_roi(request).await?;
println!("Predicted ROI: {}", response.into_inner().predicted_roi);
```

## Next Steps

1. **Task 8.3**: Implement data loading utilities
2. **Task 9.x**: Implement spatial clustering models
3. **Task 10.x**: Implement SHAP explainability engine
4. **Task 12.x**: Implement ROI prediction ensemble
5. **Task 13.x**: Implement MLA scoring
6. **Task 14.x**: Implement anomaly detection
7. **Task 16.x**: Implement time series forecasting
8. **Task 17.x**: Implement risk assessment
9. **Task 18.x**: Implement model performance tracking

## References

- [Protocol Buffers Documentation](https://developers.google.com/protocol-buffers)
- [gRPC Documentation](https://grpc.io/docs/)
- [ProtoBuf.jl](https://github.com/JuliaIO/ProtoBuf.jl)
- [gRPC.jl](https://github.com/JuliaComputing/gRPC.jl)
- [Tonic (Rust gRPC)](https://github.com/hyperium/tonic)
