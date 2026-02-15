# Analytics Engine - Julia ML/Spatial Analytics

The Analytics Engine is the AI/ML spatial analytics component of the AI-Powered Blockchain GIS Platform, implemented in Julia for high-performance numerical computing.

## Overview

This module provides:
- **Economic cluster detection** using spatial clustering (DBSCAN, K-Means)
- **Investment ROI prediction** using ensemble regression (Random Forest, XGBoost, Neural Networks)
- **MLA development impact scoring** using weighted AI indices
- **Anomaly detection** in spending using unsupervised learning (Isolation Forest, Autoencoders)
- **Explainable AI outputs** using SHAP feature importance
- **Time series forecasting** for development planning
- **Risk assessment** for investment projects

## Dependencies

### Machine Learning
- **MLJ** (0.19) - Core ML framework with unified interface
- **XGBoost** (2.4) - Gradient boosting for regression and classification
- **Flux** (0.14) - Neural networks and deep learning
- **Clustering** (0.15) - Spatial clustering algorithms (DBSCAN, K-Means)
- **OutlierDetection** (0.3) - Anomaly detection (Isolation Forest)
- **ShapML** (0.2) - SHAP explainability for model interpretability

### Database & Communication
- **LibPQ** (1.16) - PostgreSQL/PostGIS database connection
- **gRPC** (0.8) - gRPC server for communication with Rust backend
- **ProtoBuf** (1.0) - Protocol Buffers for efficient serialization

### Data Processing
- **DataFrames** (1.6) - Tabular data manipulation
- **CSV** (0.10) - CSV file I/O
- **Statistics** (1.10) - Statistical functions
- **StatsBase** (0.34) - Statistical utilities
- **Distances** (0.10) - Distance metrics for spatial analysis

### Time Series
- **TimeSeries** (0.24) - Time series data structures
- **StateSpaceModels** (0.6) - Time series forecasting

### Utilities
- **JSON3** (1.14) - JSON serialization
- **Dates** (1.10) - Date/time handling
- **Logging** (1.10) - Logging utilities

## Setup

### Prerequisites
- Julia 1.9 or higher
- PostgreSQL with PostGIS extension (for database connectivity)

### Installation

1. Navigate to the analytics directory:
```bash
cd analytics
```

2. Start Julia and activate the project:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Test dependencies:
```bash
julia test_dependencies.jl
```

### Verify Setup

Run the test script to verify all dependencies are correctly installed:
```bash
julia test_dependencies.jl
```

Expected output:
```
Testing Julia Analytics Engine Dependencies
============================================================

Instantiating project dependencies...
✓ Dependencies instantiated successfully

Testing dependency loading:
------------------------------------------------------------
✓ MLJ - Machine Learning framework
✓ XGBoost - Gradient boosting
✓ Flux - Neural networks
✓ Clustering - Spatial clustering algorithms
✓ OutlierDetection - Anomaly detection
✓ ShapML - SHAP explainability
✓ LibPQ - PostGIS database connection
✓ gRPC - gRPC communication
...
------------------------------------------------------------

✓ All dependencies loaded successfully!
```

## Module Structure

```
analytics/
├── Project.toml              # Julia project dependencies
├── README.md                 # This file
├── test_dependencies.jl      # Dependency verification script
├── Dockerfile                # Docker container configuration
└── src/
    ├── AnalyticsEngine.jl    # Main module entry point
    ├── models/               # ML model implementations (to be added)
    │   ├── roi_predictor.jl
    │   ├── anomaly_detector.jl
    │   ├── spatial_clusterer.jl
    │   ├── mla_scorer.jl
    │   ├── forecaster.jl
    │   └── risk_assessor.jl
    ├── explainability/       # SHAP explainability engine (to be added)
    │   └── shap_engine.jl
    ├── monitoring/           # Model performance tracking (to be added)
    │   └── performance_tracker.jl
    ├── utils/                # Utility functions (to be added)
    │   ├── data_loader.jl
    │   └── feature_engineering.jl
    └── grpc/                 # gRPC service implementation (to be added)
        └── service.jl
```

## Usage

### Starting the Analytics Engine

```julia
using AnalyticsEngine

# Start the gRPC service
AnalyticsEngine.start_service()
```

### Database Connection

The Analytics Engine connects to PostGIS using LibPQ for spatial data operations:

```julia
using LibPQ

# Connection string format
conn = LibPQ.Connection("host=localhost dbname=gis_platform user=postgres password=postgres")
```

### gRPC Communication

The Analytics Engine exposes a gRPC service for communication with the Rust backend:

```julia
# Service definition in Protocol Buffers (to be implemented)
# - PredictROI
# - DetectAnomalies
# - DetectClusters
# - ComputeMLAScore
# - GenerateForecast
# - AssessRisk
# - ExplainPrediction
# - GetModelMetrics
```

## Requirements Satisfied

This setup satisfies the following requirements:

- **Requirement 10.1**: Analytics Engine implemented in Julia programming language
- **Requirement 10.2**: Interface with Backend API through gRPC protocol
- **Requirement 10.3**: Execute appropriate AI/ML algorithms (regression, clustering, anomaly detection)
- **Requirement 10.4**: Fetch training data from PostGIS Geospatial Store
- **Requirement 10.5**: Support parallel computation for large-scale spatial analysis
- **Requirement 10.6**: Return results to Backend API with execution metadata
- **Requirement 10.7**: Cache trained models to avoid redundant training

## Next Steps

1. Implement gRPC service definition (Task 8.2)
2. Implement data loading utilities (Task 8.3)
3. Implement ROI prediction model (Task 9.x)
4. Implement anomaly detection model (Task 11.x)
5. Implement spatial clustering (Task 12.x)
6. Implement MLA scoring (Task 13.x)
7. Implement SHAP explainability engine (Task 10.x)

## Development

### Adding New Dependencies

To add a new dependency:

1. Edit `Project.toml` and add the package under `[deps]`
2. Add version constraint under `[compat]`
3. Run `Pkg.instantiate()` to install
4. Update this README with the new dependency

### Testing

Run the dependency test script after any changes:
```bash
julia test_dependencies.jl
```

## Performance Considerations

- Julia's JIT compilation provides near-C performance for numerical computations
- First run of functions will be slower due to compilation
- Use `@time` macro to profile performance
- Consider using `@threads` for parallel operations on large datasets
- LibPQ connection pooling for efficient database access

## References

- [MLJ Documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/)
- [Flux Documentation](https://fluxml.ai/Flux.jl/stable/)
- [ShapML Documentation](https://github.com/nredell/ShapML.jl)
- [LibPQ Documentation](https://github.com/iamed2/LibPQ.jl)
- [gRPC.jl Documentation](https://github.com/JuliaComputing/gRPC.jl)
