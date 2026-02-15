-- Migration 001: Create PostGIS database schema for AI-Powered Blockchain GIS Platform
-- This migration creates all tables with spatial indexes and foreign key constraints
-- Requirements: 11.1, 11.2, 11.3, 11.4, 11.5

-- Enable required extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- GI Products Table
-- Stores Geographical Indication products with spatial data
-- ============================================================================
CREATE TABLE gi_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    location GEOMETRY(Point, 4326) NOT NULL,
    region VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128)
);

-- Spatial index for location-based queries
CREATE INDEX idx_gi_products_location ON gi_products USING GIST(location);
-- Index for region-based filtering
CREATE INDEX idx_gi_products_region ON gi_products(region);
-- Index for category-based filtering
CREATE INDEX idx_gi_products_category ON gi_products(category);

COMMENT ON TABLE gi_products IS 'Geographical Indication products with spatial coordinates';
COMMENT ON COLUMN gi_products.location IS 'Geographic coordinates in WGS84 (SRID 4326)';
COMMENT ON COLUMN gi_products.metadata_hash IS 'SHA-256 hash of product metadata for blockchain verification';
COMMENT ON COLUMN gi_products.blockchain_tx_id IS 'Substrate blockchain transaction ID';

-- ============================================================================
-- Economic Clusters Table
-- Stores detected economic clusters with spatial boundaries
-- ============================================================================
CREATE TABLE economic_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id INTEGER NOT NULL,
    boundary GEOMETRY(Polygon, 4326) NOT NULL,
    centroid GEOMETRY(Point, 4326) NOT NULL,
    algorithm VARCHAR(50),
    parameters JSONB,
    member_count INTEGER,
    density DOUBLE PRECISION,
    economic_value DOUBLE PRECISION,
    quality_metrics JSONB,
    detected_at TIMESTAMP DEFAULT NOW(),
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128)
);

-- Spatial indexes for boundary and centroid queries
CREATE INDEX idx_clusters_boundary ON economic_clusters USING GIST(boundary);
CREATE INDEX idx_clusters_centroid ON economic_clusters USING GIST(centroid);
-- Index for algorithm-based filtering
CREATE INDEX idx_clusters_algorithm ON economic_clusters(algorithm);
-- Index for time-based queries
CREATE INDEX idx_clusters_detected_at ON economic_clusters(detected_at DESC);

COMMENT ON TABLE economic_clusters IS 'AI-detected economic clusters using DBSCAN and K-Means algorithms';
COMMENT ON COLUMN economic_clusters.boundary IS 'Cluster boundary polygon (convex hull)';
COMMENT ON COLUMN economic_clusters.quality_metrics IS 'JSON containing silhouette score, Davies-Bouldin index, Calinski-Harabasz score';

-- ============================================================================
-- ROI Predictions Table
-- Stores investment ROI predictions from ensemble models
-- ============================================================================
CREATE TABLE roi_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location GEOMETRY(Point, 4326) NOT NULL,
    sector VARCHAR(100),
    investment_amount DOUBLE PRECISION,
    timeframe_years INTEGER,
    predicted_roi DOUBLE PRECISION,
    confidence_lower DOUBLE PRECISION,
    confidence_upper DOUBLE PRECISION,
    variance DOUBLE PRECISION,
    model_version VARCHAR(50),
    features JSONB,
    shap_values JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    actual_roi DOUBLE PRECISION,
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128)
);

-- Spatial index for location-based queries
CREATE INDEX idx_predictions_location ON roi_predictions USING GIST(location);
-- Index for sector-based filtering
CREATE INDEX idx_predictions_sector ON roi_predictions(sector);
-- Index for time-based queries
CREATE INDEX idx_predictions_created_at ON roi_predictions(created_at DESC);
-- Index for model version tracking
CREATE INDEX idx_predictions_model_version ON roi_predictions(model_version);

COMMENT ON TABLE roi_predictions IS 'ROI predictions using ensemble of Random Forest, XGBoost, and Neural Network';
COMMENT ON COLUMN roi_predictions.features IS 'JSON containing input features used for prediction';
COMMENT ON COLUMN roi_predictions.shap_values IS 'JSON containing SHAP feature importance values';
COMMENT ON COLUMN roi_predictions.actual_roi IS 'Actual ROI outcome (filled in later for model performance tracking)';

-- ============================================================================
-- MLA Scores Table
-- Stores MLA development impact scores with spatial boundaries
-- ============================================================================
CREATE TABLE mla_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    constituency_id VARCHAR(100) NOT NULL,
    constituency_boundary GEOMETRY(Polygon, 4326) NOT NULL,
    overall_score DOUBLE PRECISION,
    infrastructure_score DOUBLE PRECISION,
    education_score DOUBLE PRECISION,
    healthcare_score DOUBLE PRECISION,
    employment_score DOUBLE PRECISION,
    economic_growth_score DOUBLE PRECISION,
    weights JSONB,
    indicators JSONB,
    shap_values JSONB,
    confidence_interval JSONB,
    computed_at TIMESTAMP DEFAULT NOW(),
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128)
);

-- Spatial index for boundary queries
CREATE INDEX idx_mla_scores_boundary ON mla_scores USING GIST(constituency_boundary);
-- Index for constituency-based queries
CREATE INDEX idx_mla_scores_constituency ON mla_scores(constituency_id);
-- Index for time-based queries
CREATE INDEX idx_mla_scores_computed_at ON mla_scores(computed_at DESC);
-- Index for score-based filtering
CREATE INDEX idx_mla_scores_overall ON mla_scores(overall_score DESC);

COMMENT ON TABLE mla_scores IS 'MLA development impact scores using weighted AI index with multi-criteria decision analysis';
COMMENT ON COLUMN mla_scores.weights IS 'JSON containing learned weights for each indicator category';
COMMENT ON COLUMN mla_scores.indicators IS 'JSON containing detailed development indicators';

-- ============================================================================
-- Anomalies Table
-- Stores detected anomalies in development spending
-- ============================================================================
CREATE TABLE anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(100),
    location GEOMETRY(Point, 4326),
    amount DOUBLE PRECISION,
    category VARCHAR(100),
    timestamp TIMESTAMP,
    anomaly_score DOUBLE PRECISION,
    severity VARCHAR(20),
    isolation_forest_score DOUBLE PRECISION,
    autoencoder_score DOUBLE PRECISION,
    shap_values JSONB,
    expected_value DOUBLE PRECISION,
    deviation_metrics JSONB,
    detected_at TIMESTAMP DEFAULT NOW(),
    confirmed BOOLEAN DEFAULT FALSE,
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128),
    CONSTRAINT chk_severity CHECK (severity IN ('low', 'medium', 'high', 'critical'))
);

-- Spatial index for location-based queries
CREATE INDEX idx_anomalies_location ON anomalies USING GIST(location);
-- Index for severity-based filtering
CREATE INDEX idx_anomalies_severity ON anomalies(severity);
-- Index for score-based sorting
CREATE INDEX idx_anomalies_score ON anomalies(anomaly_score DESC);
-- Index for time-based queries
CREATE INDEX idx_anomalies_detected_at ON anomalies(detected_at DESC);
-- Index for transaction lookup
CREATE INDEX idx_anomalies_transaction_id ON anomalies(transaction_id);

COMMENT ON TABLE anomalies IS 'Anomalies detected using Isolation Forest and Autoencoder algorithms';
COMMENT ON COLUMN anomalies.severity IS 'Severity level: low, medium, high, or critical';
COMMENT ON COLUMN anomalies.deviation_metrics IS 'JSON containing z-score and percentile rank';

-- ============================================================================
-- Forecasts Table
-- Stores time series forecasts for development indicators
-- ============================================================================
CREATE TABLE forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region VARCHAR(100),
    indicator VARCHAR(100),
    horizon_years INTEGER,
    forecast_values JSONB,
    model_type VARCHAR(50),
    mape DOUBLE PRECISION,
    shap_values JSONB,
    generated_at TIMESTAMP DEFAULT NOW(),
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128)
);

-- Index for region-based queries
CREATE INDEX idx_forecasts_region ON forecasts(region);
-- Index for indicator-based queries
CREATE INDEX idx_forecasts_indicator ON forecasts(indicator);
-- Index for time-based queries
CREATE INDEX idx_forecasts_generated_at ON forecasts(generated_at DESC);
-- Index for model type filtering
CREATE INDEX idx_forecasts_model_type ON forecasts(model_type);

COMMENT ON TABLE forecasts IS 'Time series forecasts using ensemble of ARIMA, Prophet, and LSTM models';
COMMENT ON COLUMN forecasts.forecast_values IS 'JSON array of {date, value, lower_bound, upper_bound}';
COMMENT ON COLUMN forecasts.mape IS 'Mean Absolute Percentage Error for forecast accuracy';

-- ============================================================================
-- Risk Assessments Table
-- Stores risk assessments for projects
-- ============================================================================
CREATE TABLE risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(100),
    location GEOMETRY(Point, 4326),
    risk_level VARCHAR(20),
    risk_probabilities JSONB,
    overall_score DOUBLE PRECISION,
    financial_risk DOUBLE PRECISION,
    operational_risk DOUBLE PRECISION,
    social_risk DOUBLE PRECISION,
    environmental_risk DOUBLE PRECISION,
    shap_values JSONB,
    mitigation_recommendations TEXT[],
    assessed_at TIMESTAMP DEFAULT NOW(),
    metadata_hash VARCHAR(64),
    blockchain_tx_id VARCHAR(128),
    CONSTRAINT chk_risk_level CHECK (risk_level IN ('low', 'medium', 'high', 'critical'))
);

-- Spatial index for location-based queries
CREATE INDEX idx_risk_location ON risk_assessments USING GIST(location);
-- Index for project lookup
CREATE INDEX idx_risk_project_id ON risk_assessments(project_id);
-- Index for risk level filtering
CREATE INDEX idx_risk_level ON risk_assessments(risk_level);
-- Index for time-based queries
CREATE INDEX idx_risk_assessed_at ON risk_assessments(assessed_at DESC);

COMMENT ON TABLE risk_assessments IS 'Multi-dimensional risk assessments for projects';
COMMENT ON COLUMN risk_assessments.risk_probabilities IS 'JSON containing calibrated probability estimates';

-- ============================================================================
-- Model Performance Table
-- Tracks AI model performance metrics and KPIs
-- ============================================================================
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value DOUBLE PRECISION,
    window_days INTEGER,
    measured_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Composite index for model performance queries
CREATE INDEX idx_model_performance_name ON model_performance(model_name, measured_at DESC);
-- Index for metric-based queries
CREATE INDEX idx_model_performance_metric ON model_performance(metric_name);
-- Index for version tracking
CREATE INDEX idx_model_performance_version ON model_performance(model_version);

COMMENT ON TABLE model_performance IS 'AI model performance tracking with KPIs (accuracy, precision, recall, F1, AUC-ROC, R-squared, RMSE, MAE)';
COMMENT ON COLUMN model_performance.window_days IS 'Rolling window period (7, 30, or 90 days)';

-- ============================================================================
-- Users Table
-- Stores user authentication and authorization data
-- ============================================================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    role VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    CONSTRAINT chk_role CHECK (role IN ('administrator', 'analyst', 'viewer'))
);

-- Index for username lookup
CREATE INDEX idx_users_username ON users(username);
-- Index for role-based queries
CREATE INDEX idx_users_role ON users(role);

COMMENT ON TABLE users IS 'User accounts with role-based access control';
COMMENT ON COLUMN users.role IS 'User role: administrator, analyst, or viewer';

-- ============================================================================
-- Audit Log Table
-- Stores audit trail of all system operations
-- ============================================================================
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100),
    entity_type VARCHAR(50),
    entity_id UUID,
    ip_address INET,
    timestamp TIMESTAMP DEFAULT NOW(),
    details JSONB
);

-- Composite index for user audit queries
CREATE INDEX idx_audit_log_user ON audit_log(user_id, timestamp DESC);
-- Composite index for entity audit queries
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
-- Index for time-based queries
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
-- Index for action-based queries
CREATE INDEX idx_audit_log_action ON audit_log(action);

COMMENT ON TABLE audit_log IS 'Comprehensive audit trail of all system operations';
COMMENT ON COLUMN audit_log.details IS 'JSON containing operation-specific details';

-- ============================================================================
-- Foreign Key Constraints
-- ============================================================================

-- Audit log references users
-- (Already defined inline above with REFERENCES)

-- ============================================================================
-- Validation Functions
-- ============================================================================

-- Function to validate Indian geographic boundaries
CREATE OR REPLACE FUNCTION validate_indian_boundaries(geom GEOMETRY)
RETURNS BOOLEAN AS $$
BEGIN
    -- India bounding box: approximately 6.5°N to 35.5°N, 68°E to 97.5°E
    -- This is a simplified check; more precise validation could use actual boundary polygons
    RETURN ST_X(geom) BETWEEN 68.0 AND 97.5 
       AND ST_Y(geom) BETWEEN 6.5 AND 35.5;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION validate_indian_boundaries IS 'Validates that coordinates fall within Indian geographic boundaries';

-- ============================================================================
-- Triggers for updated_at timestamp
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for gi_products table
CREATE TRIGGER update_gi_products_updated_at
    BEFORE UPDATE ON gi_products
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View for high-severity anomalies
CREATE VIEW high_severity_anomalies AS
SELECT 
    id,
    transaction_id,
    ST_AsText(location) as location_text,
    ST_X(location) as longitude,
    ST_Y(location) as latitude,
    amount,
    category,
    anomaly_score,
    severity,
    detected_at
FROM anomalies
WHERE severity IN ('high', 'critical')
ORDER BY anomaly_score DESC;

COMMENT ON VIEW high_severity_anomalies IS 'High and critical severity anomalies for quick monitoring';

-- View for recent model performance
CREATE VIEW recent_model_performance AS
SELECT 
    model_name,
    model_version,
    metric_name,
    metric_value,
    window_days,
    measured_at
FROM model_performance
WHERE measured_at >= NOW() - INTERVAL '30 days'
ORDER BY model_name, metric_name, measured_at DESC;

COMMENT ON VIEW recent_model_performance IS 'Model performance metrics from the last 30 days';

-- ============================================================================
-- Grant Permissions (adjust as needed for your deployment)
-- ============================================================================

-- Note: In production, create specific roles with appropriate permissions
-- This is a placeholder for development environments

-- Example: GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
-- Example: GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- ============================================================================
-- Migration Complete
-- ============================================================================

-- Verify PostGIS is working
SELECT PostGIS_Version();

-- Display table count
SELECT COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE';
