# Data Loading Utilities for Analytics Engine
# Task 8.3: Implement data loading utilities
# Requirements: 10.4

"""
Data loading utilities for fetching spatial data from PostGIS and preprocessing
it for machine learning models.

This module provides:
- Database connection management
- Spatial data extraction from PostGIS tables
- Data preprocessing and cleaning
- Feature extraction from spatial geometries
"""

module DataLoader

using LibPQ
using DataFrames
using Dates
using Logging
using JSON3
using Statistics

export DatabaseConnection, connect_db, disconnect_db, test_connection
export load_gi_products, load_economic_clusters, load_roi_predictions
export load_mla_scores, load_anomalies, load_training_data
export extract_spatial_features, validate_data

# ============================================================================
# Database Connection Management
# ============================================================================

"""
    DatabaseConnection

Manages PostgreSQL/PostGIS database connection with connection pooling support.

# Fields
- `conn::LibPQ.Connection`: Active database connection
- `host::String`: Database host
- `port::Int`: Database port
- `dbname::String`: Database name
- `connected::Bool`: Connection status
"""
mutable struct DatabaseConnection
    conn::Union{LibPQ.Connection, Nothing}
    host::String
    port::Int
    dbname::String
    user::String
    connected::Bool
end

"""
    connect_db(; host="localhost", port=5432, dbname="gis_platform", 
               user="postgres", password="postgres")

Create a connection to the PostGIS database.

# Arguments
- `host::String`: Database host (default: "localhost")
- `port::Int`: Database port (default: 5432)
- `dbname::String`: Database name (default: "gis_platform")
- `user::String`: Database user (default: "postgres")
- `password::String`: Database password (default: "postgres")

# Returns
- `DatabaseConnection`: Connection object

# Example
```julia
db = connect_db(host="localhost", dbname="gis_platform")
```
"""
function connect_db(; 
    host::String="localhost", 
    port::Int=5432, 
    dbname::String="gis_platform",
    user::String="postgres",
    password::String="postgres"
)
    try
        conn_str = "host=$host port=$port dbname=$dbname user=$user password=$password"
        conn = LibPQ.Connection(conn_str)
        
        @info "Connected to PostGIS database" host=host dbname=dbname
        
        return DatabaseConnection(conn, host, port, dbname, user, true)
    catch e
        @error "Failed to connect to database" exception=e
        return DatabaseConnection(nothing, host, port, dbname, user, false)
    end
end

"""
    disconnect_db(db::DatabaseConnection)

Close the database connection.

# Arguments
- `db::DatabaseConnection`: Database connection to close
"""
function disconnect_db(db::DatabaseConnection)
    if db.connected && db.conn !== nothing
        try
            close(db.conn)
            db.connected = false
            @info "Database connection closed"
        catch e
            @error "Error closing database connection" exception=e
        end
    end
end

"""
    test_connection(db::DatabaseConnection)

Test if the database connection is active and PostGIS is available.

# Arguments
- `db::DatabaseConnection`: Database connection to test

# Returns
- `Bool`: true if connection is active and PostGIS is available
"""
function test_connection(db::DatabaseConnection)
    if !db.connected || db.conn === nothing
        @warn "Database not connected"
        return false
    end
    
    try
        # Test basic query
        result = execute(db.conn, "SELECT 1 as test")
        df = DataFrame(result)
        
        # Test PostGIS extension
        result = execute(db.conn, "SELECT PostGIS_Version() as version")
        postgis_df = DataFrame(result)
        
        @info "Database connection test successful" postgis_version=postgis_df[1, :version]
        return true
    catch e
        @error "Database connection test failed" exception=e
        return false
    end
end

# ============================================================================
# Data Loading Functions
# ============================================================================

"""
    load_gi_products(db::DatabaseConnection; region=nothing, category=nothing, limit=nothing)

Load GI products from the database with optional filtering.

# Arguments
- `db::DatabaseConnection`: Database connection
- `region::Union{String, Nothing}`: Filter by region (optional)
- `category::Union{String, Nothing}`: Filter by category (optional)
- `limit::Union{Int, Nothing}`: Limit number of results (optional)

# Returns
- `DataFrame`: GI products with columns: id, name, description, category, 
               longitude, latitude, region, created_at, updated_at

# Example
```julia
db = connect_db()
products = load_gi_products(db, region="Karnataka", limit=100)
```
"""
function load_gi_products(
    db::DatabaseConnection;
    region::Union{String, Nothing}=nothing,
    category::Union{String, Nothing}=nothing,
    limit::Union{Int, Nothing}=nothing
)
    if !db.connected
        error("Database not connected")
    end
    
    # Build query with optional filters
    query = """
        SELECT 
            id,
            name,
            description,
            category,
            ST_X(location) as longitude,
            ST_Y(location) as latitude,
            region,
            created_at,
            updated_at
        FROM gi_products
        WHERE 1=1
    """
    
    if region !== nothing
        query *= " AND region = '$region'"
    end
    
    if category !== nothing
        query *= " AND category = '$category'"
    end
    
    query *= " ORDER BY created_at DESC"
    
    if limit !== nothing
        query *= " LIMIT $limit"
    end
    
    try
        result = execute(db.conn, query)
        df = DataFrame(result)
        @info "Loaded GI products" count=nrow(df)
        return df
    catch e
        @error "Failed to load GI products" exception=e
        rethrow(e)
    end
end

"""
    load_economic_clusters(db::DatabaseConnection; algorithm=nothing, min_quality=nothing, limit=nothing)

Load economic clusters from the database with optional filtering.

# Arguments
- `db::DatabaseConnection`: Database connection
- `algorithm::Union{String, Nothing}`: Filter by algorithm (e.g., "DBSCAN", "K-Means")
- `min_quality::Union{Float64, Nothing}`: Minimum silhouette score
- `limit::Union{Int, Nothing}`: Limit number of results

# Returns
- `DataFrame`: Economic clusters with spatial and quality metrics
"""
function load_economic_clusters(
    db::DatabaseConnection;
    algorithm::Union{String, Nothing}=nothing,
    min_quality::Union{Float64, Nothing}=nothing,
    limit::Union{Int, Nothing}=nothing
)
    if !db.connected
        error("Database not connected")
    end
    
    query = """
        SELECT 
            id,
            cluster_id,
            ST_AsText(boundary) as boundary_wkt,
            ST_X(centroid) as centroid_lon,
            ST_Y(centroid) as centroid_lat,
            algorithm,
            parameters,
            member_count,
            density,
            economic_value,
            quality_metrics,
            detected_at
        FROM economic_clusters
        WHERE 1=1
    """
    
    if algorithm !== nothing
        query *= " AND algorithm = '$algorithm'"
    end
    
    if min_quality !== nothing
        query *= " AND (quality_metrics->>'silhouette_score')::float >= $min_quality"
    end
    
    query *= " ORDER BY detected_at DESC"
    
    if limit !== nothing
        query *= " LIMIT $limit"
    end
    
    try
        result = execute(db.conn, query)
        df = DataFrame(result)
        @info "Loaded economic clusters" count=nrow(df)
        return df
    catch e
        @error "Failed to load economic clusters" exception=e
        rethrow(e)
    end
end

"""
    load_roi_predictions(db::DatabaseConnection; sector=nothing, min_date=nothing, limit=nothing)

Load ROI predictions from the database with optional filtering.

# Arguments
- `db::DatabaseConnection`: Database connection
- `sector::Union{String, Nothing}`: Filter by sector
- `min_date::Union{DateTime, Nothing}`: Filter predictions after this date
- `limit::Union{Int, Nothing}`: Limit number of results

# Returns
- `DataFrame`: ROI predictions with features and SHAP values
"""
function load_roi_predictions(
    db::DatabaseConnection;
    sector::Union{String, Nothing}=nothing,
    min_date::Union{DateTime, Nothing}=nothing,
    limit::Union{Int, Nothing}=nothing
)
    if !db.connected
        error("Database not connected")
    end
    
    query = """
        SELECT 
            id,
            ST_X(location) as longitude,
            ST_Y(location) as latitude,
            sector,
            investment_amount,
            timeframe_years,
            predicted_roi,
            confidence_lower,
            confidence_upper,
            variance,
            model_version,
            features,
            shap_values,
            created_at,
            actual_roi
        FROM roi_predictions
        WHERE 1=1
    """
    
    if sector !== nothing
        query *= " AND sector = '$sector'"
    end
    
    if min_date !== nothing
        date_str = Dates.format(min_date, "yyyy-mm-dd HH:MM:SS")
        query *= " AND created_at >= '$date_str'"
    end
    
    query *= " ORDER BY created_at DESC"
    
    if limit !== nothing
        query *= " LIMIT $limit"
    end
    
    try
        result = execute(db.conn, query)
        df = DataFrame(result)
        @info "Loaded ROI predictions" count=nrow(df)
        return df
    catch e
        @error "Failed to load ROI predictions" exception=e
        rethrow(e)
    end
end

"""
    load_mla_scores(db::DatabaseConnection; constituency_id=nothing, min_score=nothing, limit=nothing)

Load MLA development impact scores from the database.

# Arguments
- `db::DatabaseConnection`: Database connection
- `constituency_id::Union{String, Nothing}`: Filter by constituency ID
- `min_score::Union{Float64, Nothing}`: Minimum overall score
- `limit::Union{Int, Nothing}`: Limit number of results

# Returns
- `DataFrame`: MLA scores with indicators and SHAP values
"""
function load_mla_scores(
    db::DatabaseConnection;
    constituency_id::Union{String, Nothing}=nothing,
    min_score::Union{Float64, Nothing}=nothing,
    limit::Union{Int, Nothing}=nothing
)
    if !db.connected
        error("Database not connected")
    end
    
    query = """
        SELECT 
            id,
            constituency_id,
            ST_AsText(constituency_boundary) as boundary_wkt,
            overall_score,
            infrastructure_score,
            education_score,
            healthcare_score,
            employment_score,
            economic_growth_score,
            weights,
            indicators,
            shap_values,
            confidence_interval,
            computed_at
        FROM mla_scores
        WHERE 1=1
    """
    
    if constituency_id !== nothing
        query *= " AND constituency_id = '$constituency_id'"
    end
    
    if min_score !== nothing
        query *= " AND overall_score >= $min_score"
    end
    
    query *= " ORDER BY computed_at DESC"
    
    if limit !== nothing
        query *= " LIMIT $limit"
    end
    
    try
        result = execute(db.conn, query)
        df = DataFrame(result)
        @info "Loaded MLA scores" count=nrow(df)
        return df
    catch e
        @error "Failed to load MLA scores" exception=e
        rethrow(e)
    end
end

"""
    load_anomalies(db::DatabaseConnection; min_score=nothing, severity=nothing, limit=nothing)

Load detected anomalies from the database.

# Arguments
- `db::DatabaseConnection`: Database connection
- `min_score::Union{Float64, Nothing}`: Minimum anomaly score
- `severity::Union{String, Nothing}`: Filter by severity ("low", "medium", "high", "critical")
- `limit::Union{Int, Nothing}`: Limit number of results

# Returns
- `DataFrame`: Anomalies with scores and SHAP explanations
"""
function load_anomalies(
    db::DatabaseConnection;
    min_score::Union{Float64, Nothing}=nothing,
    severity::Union{String, Nothing}=nothing,
    limit::Union{Int, Nothing}=nothing
)
    if !db.connected
        error("Database not connected")
    end
    
    query = """
        SELECT 
            id,
            transaction_id,
            ST_X(location) as longitude,
            ST_Y(location) as latitude,
            amount,
            category,
            timestamp,
            anomaly_score,
            severity,
            isolation_forest_score,
            autoencoder_score,
            shap_values,
            expected_value,
            deviation_metrics,
            detected_at,
            confirmed
        FROM anomalies
        WHERE 1=1
    """
    
    if min_score !== nothing
        query *= " AND anomaly_score >= $min_score"
    end
    
    if severity !== nothing
        query *= " AND severity = '$severity'"
    end
    
    query *= " ORDER BY anomaly_score DESC, detected_at DESC"
    
    if limit !== nothing
        query *= " LIMIT $limit"
    end
    
    try
        result = execute(db.conn, query)
        df = DataFrame(result)
        @info "Loaded anomalies" count=nrow(df)
        return df
    catch e
        @error "Failed to load anomalies" exception=e
        rethrow(e)
    end
end

"""
    load_training_data(db::DatabaseConnection, table::String, features::Vector{String}; 
                       target=nothing, where_clause=nothing, limit=nothing)

Generic function to load training data from any table with specified features.

# Arguments
- `db::DatabaseConnection`: Database connection
- `table::String`: Table name
- `features::Vector{String}`: List of feature column names
- `target::Union{String, Nothing}`: Target column name (optional)
- `where_clause::Union{String, Nothing}`: SQL WHERE clause (optional)
- `limit::Union{Int, Nothing}`: Limit number of results

# Returns
- `DataFrame`: Training data with specified features and optional target

# Example
```julia
data = load_training_data(
    db, 
    "roi_predictions",
    ["investment_amount", "timeframe_years", "sector"],
    target="actual_roi",
    where_clause="actual_roi IS NOT NULL",
    limit=1000
)
```
"""
function load_training_data(
    db::DatabaseConnection,
    table::String,
    features::Vector{String};
    target::Union{String, Nothing}=nothing,
    where_clause::Union{String, Nothing}=nothing,
    limit::Union{Int, Nothing}=nothing
)
    if !db.connected
        error("Database not connected")
    end
    
    # Build column list
    columns = join(features, ", ")
    if target !== nothing
        columns *= ", $target"
    end
    
    query = "SELECT $columns FROM $table"
    
    if where_clause !== nothing
        query *= " WHERE $where_clause"
    end
    
    if limit !== nothing
        query *= " LIMIT $limit"
    end
    
    try
        result = execute(db.conn, query)
        df = DataFrame(result)
        @info "Loaded training data" table=table features=length(features) rows=nrow(df)
        return df
    catch e
        @error "Failed to load training data" table=table exception=e
        rethrow(e)
    end
end

# ============================================================================
# Spatial Feature Extraction
# ============================================================================

"""
    extract_spatial_features(df::DataFrame; lon_col="longitude", lat_col="latitude")

Extract spatial features from geographic coordinates.

# Arguments
- `df::DataFrame`: DataFrame with longitude and latitude columns
- `lon_col::String`: Name of longitude column (default: "longitude")
- `lat_col::String`: Name of latitude column (default: "latitude")

# Returns
- `DataFrame`: Original DataFrame with additional spatial features:
  - `distance_to_center`: Distance to geographic center (km)
  - `region_quadrant`: Geographic quadrant (NE, NW, SE, SW)

# Example
```julia
df = load_gi_products(db)
df_with_features = extract_spatial_features(df)
```
"""
function extract_spatial_features(
    df::DataFrame;
    lon_col::String="longitude",
    lat_col::String="latitude"
)
    if nrow(df) == 0
        @warn "Empty DataFrame provided"
        return df
    end
    
    # Calculate geographic center
    center_lon = mean(df[!, lon_col])
    center_lat = mean(df[!, lat_col])
    
    # Calculate distance to center (simplified Euclidean distance in degrees)
    # For more accurate distance, use Haversine formula
    df[!, :distance_to_center] = sqrt.((df[!, lon_col] .- center_lon).^2 .+ 
                                       (df[!, lat_col] .- center_lat).^2) .* 111.0  # Approx km per degree
    
    # Determine quadrant
    df[!, :region_quadrant] = map(df[!, lon_col], df[!, lat_col]) do lon, lat
        if lon >= center_lon && lat >= center_lat
            "NE"
        elseif lon < center_lon && lat >= center_lat
            "NW"
        elseif lon < center_lon && lat < center_lat
            "SW"
        else
            "SE"
        end
    end
    
    @info "Extracted spatial features" center_lon=center_lon center_lat=center_lat
    return df
end

"""
    validate_data(df::DataFrame; required_cols::Vector{String}=String[], 
                  check_missing::Bool=true, check_types::Bool=true)

Validate DataFrame for ML model training.

# Arguments
- `df::DataFrame`: DataFrame to validate
- `required_cols::Vector{String}`: Required column names
- `check_missing::Bool`: Check for missing values (default: true)
- `check_types::Bool`: Check for numeric types (default: true)

# Returns
- `Tuple{Bool, Vector{String}}`: (is_valid, error_messages)

# Example
```julia
valid, errors = validate_data(df, required_cols=["longitude", "latitude", "value"])
if !valid
    @error "Data validation failed" errors=errors
end
```
"""
function validate_data(
    df::DataFrame;
    required_cols::Vector{String}=String[],
    check_missing::Bool=true,
    check_types::Bool=true
)
    errors = String[]
    
    # Check if DataFrame is empty
    if nrow(df) == 0
        push!(errors, "DataFrame is empty")
        return (false, errors)
    end
    
    # Check required columns
    for col in required_cols
        if !(col in names(df))
            push!(errors, "Missing required column: $col")
        end
    end
    
    # Check for missing values
    if check_missing
        for col in names(df)
            missing_count = count(ismissing, df[!, col])
            if missing_count > 0
                push!(errors, "Column '$col' has $missing_count missing values")
            end
        end
    end
    
    # Check for numeric types in numeric columns
    if check_types
        for col in names(df)
            if col in ["longitude", "latitude", "amount", "score", "value"]
                if !all(x -> ismissing(x) || x isa Number, df[!, col])
                    push!(errors, "Column '$col' contains non-numeric values")
                end
            end
        end
    end
    
    is_valid = length(errors) == 0
    
    if is_valid
        @info "Data validation passed" rows=nrow(df) cols=ncol(df)
    else
        @warn "Data validation failed" error_count=length(errors)
    end
    
    return (is_valid, errors)
end

end # module DataLoader
