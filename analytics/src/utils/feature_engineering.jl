# Feature Engineering Utilities for Analytics Engine
# Task 8.3: Implement data loading utilities
# Requirements: 10.4

"""
Feature engineering utilities for preprocessing spatial data and creating
features for machine learning models.

This module provides:
- Data normalization and scaling
- Categorical encoding
- Temporal feature extraction
- Spatial feature engineering
- Feature selection and dimensionality reduction
"""

module FeatureEngineering

using DataFrames
using Statistics
using StatsBase
using Dates
using Logging

export normalize_features, standardize_features, encode_categorical
export extract_temporal_features, create_spatial_bins
export handle_missing_values, remove_outliers, select_features
export create_interaction_features, polynomial_features

# ============================================================================
# Data Normalization and Scaling
# ============================================================================

"""
    normalize_features(df::DataFrame, cols::Vector{String}; method=:minmax)

Normalize numeric features to [0, 1] range using min-max scaling.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Vector{String}`: Columns to normalize
- `method::Symbol`: Normalization method (:minmax or :zscore)

# Returns
- `DataFrame`: DataFrame with normalized features
- `Dict`: Normalization parameters for inverse transform

# Example
```julia
df_norm, params = normalize_features(df, ["amount", "distance"])
```
"""
function normalize_features(
    df::DataFrame,
    cols::Vector{String};
    method::Symbol=:minmax
)
    df_copy = copy(df)
    params = Dict{String, Any}()
    
    for col in cols
        if !(col in names(df))
            @warn "Column not found, skipping" col=col
            continue
        end
        
        values = df[!, col]
        
        # Skip if not numeric
        if !all(x -> ismissing(x) || x isa Number, values)
            @warn "Column is not numeric, skipping" col=col
            continue
        end
        
        # Remove missing values for calculation
        non_missing = filter(!ismissing, values)
        
        if length(non_missing) == 0
            @warn "Column has no non-missing values, skipping" col=col
            continue
        end
        
        if method == :minmax
            min_val = minimum(non_missing)
            max_val = maximum(non_missing)
            
            if min_val == max_val
                @warn "Column has constant value, setting to 0.5" col=col
                df_copy[!, col] = fill(0.5, nrow(df))
            else
                df_copy[!, col] = (values .- min_val) ./ (max_val - min_val)
            end
            
            params[col] = Dict("method" => "minmax", "min" => min_val, "max" => max_val)
            
        elseif method == :zscore
            mean_val = mean(non_missing)
            std_val = std(non_missing)
            
            if std_val == 0
                @warn "Column has zero standard deviation, setting to 0" col=col
                df_copy[!, col] = fill(0.0, nrow(df))
            else
                df_copy[!, col] = (values .- mean_val) ./ std_val
            end
            
            params[col] = Dict("method" => "zscore", "mean" => mean_val, "std" => std_val)
        else
            error("Unknown normalization method: $method")
        end
    end
    
    @info "Normalized features" method=method columns=length(cols)
    return df_copy, params
end

"""
    standardize_features(df::DataFrame, cols::Vector{String})

Standardize features to have zero mean and unit variance (z-score normalization).

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Vector{String}`: Columns to standardize

# Returns
- `DataFrame`: DataFrame with standardized features
- `Dict`: Standardization parameters

# Example
```julia
df_std, params = standardize_features(df, ["feature1", "feature2"])
```
"""
function standardize_features(df::DataFrame, cols::Vector{String})
    return normalize_features(df, cols, method=:zscore)
end

# ============================================================================
# Categorical Encoding
# ============================================================================

"""
    encode_categorical(df::DataFrame, col::String; method=:onehot, drop_first=false)

Encode categorical variables using one-hot encoding or label encoding.

# Arguments
- `df::DataFrame`: Input DataFrame
- `col::String`: Column to encode
- `method::Symbol`: Encoding method (:onehot or :label)
- `drop_first::Bool`: Drop first category to avoid multicollinearity (default: false)

# Returns
- `DataFrame`: DataFrame with encoded features
- `Dict`: Encoding mapping for inverse transform

# Example
```julia
df_encoded, mapping = encode_categorical(df, "sector", method=:onehot)
```
"""
function encode_categorical(
    df::DataFrame,
    col::String;
    method::Symbol=:onehot,
    drop_first::Bool=false
)
    if !(col in names(df))
        error("Column not found: $col")
    end
    
    df_copy = copy(df)
    
    # Get unique categories
    categories = unique(skipmissing(df[!, col]))
    categories = sort(collect(categories))
    
    if method == :onehot
        # Create one-hot encoded columns
        start_idx = drop_first ? 2 : 1
        
        for (i, cat) in enumerate(categories[start_idx:end])
            new_col = "$(col)_$(cat)"
            df_copy[!, new_col] = Int.(df[!, col] .== cat)
        end
        
        # Remove original column
        select!(df_copy, Not(col))
        
        mapping = Dict(
            "method" => "onehot",
            "categories" => categories,
            "drop_first" => drop_first
        )
        
        @info "One-hot encoded categorical feature" col=col categories=length(categories)
        
    elseif method == :label
        # Create label encoding
        cat_to_label = Dict(cat => i for (i, cat) in enumerate(categories))
        df_copy[!, col] = [get(cat_to_label, val, missing) for val in df[!, col]]
        
        mapping = Dict(
            "method" => "label",
            "categories" => categories,
            "mapping" => cat_to_label
        )
        
        @info "Label encoded categorical feature" col=col categories=length(categories)
    else
        error("Unknown encoding method: $method")
    end
    
    return df_copy, mapping
end

# ============================================================================
# Temporal Feature Extraction
# ============================================================================

"""
    extract_temporal_features(df::DataFrame, date_col::String)

Extract temporal features from datetime column.

# Arguments
- `df::DataFrame`: Input DataFrame
- `date_col::String`: Name of datetime column

# Returns
- `DataFrame`: DataFrame with additional temporal features:
  - year, month, day, day_of_week, day_of_year, quarter, is_weekend

# Example
```julia
df_temporal = extract_temporal_features(df, "created_at")
```
"""
function extract_temporal_features(df::DataFrame, date_col::String)
    if !(date_col in names(df))
        error("Date column not found: $date_col")
    end
    
    df_copy = copy(df)
    
    # Extract temporal components
    df_copy[!, :year] = year.(df[!, date_col])
    df_copy[!, :month] = month.(df[!, date_col])
    df_copy[!, :day] = day.(df[!, date_col])
    df_copy[!, :day_of_week] = dayofweek.(df[!, date_col])
    df_copy[!, :day_of_year] = dayofyear.(df[!, date_col])
    df_copy[!, :quarter] = quarterofyear.(df[!, date_col])
    df_copy[!, :is_weekend] = Int.(dayofweek.(df[!, date_col]) .>= 6)
    
    # Calculate days since epoch (useful for time-based models)
    epoch = DateTime(2020, 1, 1)
    df_copy[!, :days_since_epoch] = Dates.value.(df[!, date_col] .- epoch) ./ (1000 * 60 * 60 * 24)
    
    @info "Extracted temporal features" date_col=date_col features=8
    return df_copy
end

# ============================================================================
# Spatial Feature Engineering
# ============================================================================

"""
    create_spatial_bins(df::DataFrame, lon_col::String, lat_col::String; 
                        n_bins_lon=10, n_bins_lat=10)

Create spatial bins (grid cells) for geographic coordinates.

# Arguments
- `df::DataFrame`: Input DataFrame
- `lon_col::String`: Longitude column name
- `lat_col::String`: Latitude column name
- `n_bins_lon::Int`: Number of longitude bins (default: 10)
- `n_bins_lat::Int`: Number of latitude bins (default: 10)

# Returns
- `DataFrame`: DataFrame with spatial bin assignments

# Example
```julia
df_binned = create_spatial_bins(df, "longitude", "latitude", n_bins_lon=20, n_bins_lat=20)
```
"""
function create_spatial_bins(
    df::DataFrame,
    lon_col::String,
    lat_col::String;
    n_bins_lon::Int=10,
    n_bins_lat::Int=10
)
    if !(lon_col in names(df)) || !(lat_col in names(df))
        error("Coordinate columns not found")
    end
    
    df_copy = copy(df)
    
    # Get coordinate ranges
    lon_min, lon_max = extrema(skipmissing(df[!, lon_col]))
    lat_min, lat_max = extrema(skipmissing(df[!, lat_col]))
    
    # Create bin edges
    lon_edges = range(lon_min, lon_max, length=n_bins_lon + 1)
    lat_edges = range(lat_min, lat_max, length=n_bins_lat + 1)
    
    # Assign bins
    df_copy[!, :lon_bin] = [findfirst(x -> lon < x, lon_edges[2:end]) for lon in df[!, lon_col]]
    df_copy[!, :lat_bin] = [findfirst(x -> lat < x, lat_edges[2:end]) for lat in df[!, lat_col]]
    
    # Create combined spatial bin ID
    df_copy[!, :spatial_bin_id] = string.(df_copy[!, :lon_bin], "_", df_copy[!, :lat_bin])
    
    @info "Created spatial bins" n_bins_lon=n_bins_lon n_bins_lat=n_bins_lat unique_bins=length(unique(df_copy[!, :spatial_bin_id]))
    return df_copy
end

# ============================================================================
# Missing Value Handling
# ============================================================================

"""
    handle_missing_values(df::DataFrame, cols::Vector{String}; strategy=:mean)

Handle missing values in specified columns.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Vector{String}`: Columns to process
- `strategy::Symbol`: Imputation strategy (:mean, :median, :mode, :drop, :zero)

# Returns
- `DataFrame`: DataFrame with missing values handled

# Example
```julia
df_clean = handle_missing_values(df, ["amount", "distance"], strategy=:median)
```
"""
function handle_missing_values(
    df::DataFrame,
    cols::Vector{String};
    strategy::Symbol=:mean
)
    df_copy = copy(df)
    
    for col in cols
        if !(col in names(df))
            @warn "Column not found, skipping" col=col
            continue
        end
        
        missing_count = count(ismissing, df[!, col])
        if missing_count == 0
            continue
        end
        
        if strategy == :drop
            # Drop rows with missing values
            df_copy = df_copy[.!ismissing.(df_copy[!, col]), :]
            
        elseif strategy == :zero
            # Replace with zero
            df_copy[!, col] = coalesce.(df_copy[!, col], 0)
            
        elseif strategy == :mean
            # Replace with mean
            non_missing = filter(!ismissing, df[!, col])
            if length(non_missing) > 0 && all(x -> x isa Number, non_missing)
                mean_val = mean(non_missing)
                df_copy[!, col] = coalesce.(df_copy[!, col], mean_val)
            end
            
        elseif strategy == :median
            # Replace with median
            non_missing = filter(!ismissing, df[!, col])
            if length(non_missing) > 0 && all(x -> x isa Number, non_missing)
                median_val = median(non_missing)
                df_copy[!, col] = coalesce.(df_copy[!, col], median_val)
            end
            
        elseif strategy == :mode
            # Replace with mode (most frequent value)
            non_missing = filter(!ismissing, df[!, col])
            if length(non_missing) > 0
                mode_val = mode(non_missing)
                df_copy[!, col] = coalesce.(df_copy[!, col], mode_val)
            end
        else
            error("Unknown strategy: $strategy")
        end
        
        @info "Handled missing values" col=col strategy=strategy missing_count=missing_count
    end
    
    return df_copy
end

# ============================================================================
# Outlier Detection and Removal
# ============================================================================

"""
    remove_outliers(df::DataFrame, cols::Vector{String}; method=:iqr, threshold=1.5)

Remove outliers from specified columns.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Vector{String}`: Columns to check for outliers
- `method::Symbol`: Detection method (:iqr or :zscore)
- `threshold::Float64`: Threshold for outlier detection (default: 1.5 for IQR, 3.0 for z-score)

# Returns
- `DataFrame`: DataFrame with outliers removed

# Example
```julia
df_clean = remove_outliers(df, ["amount", "distance"], method=:iqr, threshold=1.5)
```
"""
function remove_outliers(
    df::DataFrame,
    cols::Vector{String};
    method::Symbol=:iqr,
    threshold::Float64=1.5
)
    df_copy = copy(df)
    mask = trues(nrow(df))
    
    for col in cols
        if !(col in names(df))
            @warn "Column not found, skipping" col=col
            continue
        end
        
        values = filter(!ismissing, df[!, col])
        
        if !all(x -> x isa Number, values)
            @warn "Column is not numeric, skipping" col=col
            continue
        end
        
        if method == :iqr
            # Interquartile range method
            q1 = quantile(values, 0.25)
            q3 = quantile(values, 0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            col_mask = (df[!, col] .>= lower_bound) .& (df[!, col] .<= upper_bound)
            mask .&= coalesce.(col_mask, true)
            
            outliers_removed = count(.!col_mask)
            @info "Removed outliers using IQR" col=col outliers=outliers_removed lower=lower_bound upper=upper_bound
            
        elseif method == :zscore
            # Z-score method
            mean_val = mean(values)
            std_val = std(values)
            
            if std_val > 0
                z_scores = abs.((df[!, col] .- mean_val) ./ std_val)
                col_mask = z_scores .<= threshold
                mask .&= coalesce.(col_mask, true)
                
                outliers_removed = count(.!col_mask)
                @info "Removed outliers using z-score" col=col outliers=outliers_removed threshold=threshold
            end
        else
            error("Unknown method: $method")
        end
    end
    
    df_copy = df_copy[mask, :]
    total_removed = nrow(df) - nrow(df_copy)
    @info "Total rows removed" count=total_removed
    
    return df_copy
end

# ============================================================================
# Feature Selection
# ============================================================================

"""
    select_features(df::DataFrame, target_col::String, n_features::Int; method=:correlation)

Select top N features based on correlation with target variable.

# Arguments
- `df::DataFrame`: Input DataFrame
- `target_col::String`: Target variable column name
- `n_features::Int`: Number of features to select
- `method::Symbol`: Selection method (:correlation)

# Returns
- `Vector{String}`: Selected feature names

# Example
```julia
selected = select_features(df, "actual_roi", 10, method=:correlation)
```
"""
function select_features(
    df::DataFrame,
    target_col::String,
    n_features::Int;
    method::Symbol=:correlation
)
    if !(target_col in names(df))
        error("Target column not found: $target_col")
    end
    
    # Get numeric columns (excluding target)
    numeric_cols = String[]
    for col in names(df)
        if col != target_col && all(x -> ismissing(x) || x isa Number, df[!, col])
            push!(numeric_cols, col)
        end
    end
    
    if length(numeric_cols) == 0
        @warn "No numeric features found"
        return String[]
    end
    
    if method == :correlation
        # Calculate correlation with target
        target_values = filter(!ismissing, df[!, target_col])
        
        correlations = Dict{String, Float64}()
        for col in numeric_cols
            col_values = df[!, col]
            
            # Remove rows with missing values in either column
            valid_mask = .!ismissing.(col_values) .& .!ismissing.(df[!, target_col])
            
            if count(valid_mask) > 1
                corr = cor(col_values[valid_mask], df[valid_mask, target_col])
                correlations[col] = abs(corr)
            end
        end
        
        # Sort by absolute correlation
        sorted_features = sort(collect(correlations), by=x->x[2], rev=true)
        selected = [feat for (feat, corr) in sorted_features[1:min(n_features, length(sorted_features))]]
        
        @info "Selected features by correlation" n_selected=length(selected)
        return selected
    else
        error("Unknown method: $method")
    end
end

# ============================================================================
# Feature Interaction and Polynomial Features
# ============================================================================

"""
    create_interaction_features(df::DataFrame, col1::String, col2::String)

Create interaction features between two columns.

# Arguments
- `df::DataFrame`: Input DataFrame
- `col1::String`: First column name
- `col2::String`: Second column name

# Returns
- `DataFrame`: DataFrame with interaction feature added

# Example
```julia
df_interact = create_interaction_features(df, "amount", "distance")
```
"""
function create_interaction_features(df::DataFrame, col1::String, col2::String)
    if !(col1 in names(df)) || !(col2 in names(df))
        error("Columns not found")
    end
    
    df_copy = copy(df)
    interaction_col = "$(col1)_x_$(col2)"
    df_copy[!, interaction_col] = df[!, col1] .* df[!, col2]
    
    @info "Created interaction feature" feature=interaction_col
    return df_copy
end

"""
    polynomial_features(df::DataFrame, cols::Vector{String}, degree::Int=2)

Create polynomial features up to specified degree.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Vector{String}`: Columns to create polynomial features for
- `degree::Int`: Maximum polynomial degree (default: 2)

# Returns
- `DataFrame`: DataFrame with polynomial features added

# Example
```julia
df_poly = polynomial_features(df, ["amount", "distance"], degree=2)
```
"""
function polynomial_features(df::DataFrame, cols::Vector{String}, degree::Int=2)
    df_copy = copy(df)
    
    for col in cols
        if !(col in names(df))
            @warn "Column not found, skipping" col=col
            continue
        end
        
        for d in 2:degree
            new_col = "$(col)_pow$(d)"
            df_copy[!, new_col] = df[!, col] .^ d
            @info "Created polynomial feature" feature=new_col degree=d
        end
    end
    
    return df_copy
end

end # module FeatureEngineering
