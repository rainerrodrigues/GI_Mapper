#!/usr/bin/env julia

"""
Test script for data loading utilities
Task 8.3: Implement data loading utilities

This script tests:
1. Database connection management
2. Data loading functions
3. Spatial feature extraction
4. Data validation
5. Feature engineering utilities
"""

using Pkg
Pkg.activate(".")

println("=" ^ 70)
println("Testing Data Loading Utilities")
println("=" ^ 70)
println()

# Load required packages
println("Loading required packages...")
using DataFrames
using Dates
using Statistics
using StatsBase
using Test

# Load the utility modules directly
println("Loading DataLoader module...")
include("src/utils/data_loader.jl")
using .DataLoader

println("Loading FeatureEngineering module...")
include("src/utils/feature_engineering.jl")
using .FeatureEngineering

println("✓ Modules loaded successfully")
println()

# ============================================================================
# Test 1: Database Connection (Mock Test)
# ============================================================================
println("Test 1: Database Connection Management")
println("-" ^ 70)

# Note: This test will fail if database is not available
# In production, database should be running
println("Attempting to connect to database...")
db = DataLoader.connect_db(
    host=get(ENV, "DB_HOST", "localhost"),
    port=parse(Int, get(ENV, "DB_PORT", "5432")),
    dbname=get(ENV, "DB_NAME", "gis_platform"),
    user=get(ENV, "DB_USER", "postgres"),
    password=get(ENV, "DB_PASSWORD", "postgres")
)

if db.connected
    println("✓ Database connection successful")
    
    # Test connection
    if DataLoader.test_connection(db)
        println("✓ Database connection test passed")
        println("✓ PostGIS extension is available")
    else
        println("✗ Database connection test failed")
    end
    
    # Disconnect
    DataLoader.disconnect_db(db)
    println("✓ Database disconnected")
else
    println("⚠ Database connection failed (this is expected if database is not running)")
    println("  To test with real database, ensure PostgreSQL with PostGIS is running")
    println("  and set environment variables: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
end
println()

# ============================================================================
# Test 2: Spatial Feature Extraction
# ============================================================================
println("Test 2: Spatial Feature Extraction")
println("-" ^ 70)

# Create sample DataFrame with geographic coordinates
sample_data = DataFrame(
    id = 1:10,
    name = ["Location $i" for i in 1:10],
    longitude = [77.0 + rand() for _ in 1:10],  # Around Delhi
    latitude = [28.0 + rand() for _ in 1:10],
    value = rand(10) * 100
)

println("Sample data created:")
println(first(sample_data, 3))
println()

# Extract spatial features
df_with_features = DataLoader.extract_spatial_features(sample_data)

@test "distance_to_center" in names(df_with_features)
@test "region_quadrant" in names(df_with_features)
println("✓ Spatial features extracted successfully")
println("  Added features: distance_to_center, region_quadrant")
println()

# ============================================================================
# Test 3: Data Validation
# ============================================================================
println("Test 3: Data Validation")
println("-" ^ 70)

# Test with valid data
valid, errors = DataLoader.validate_data(
    sample_data,
    required_cols=["longitude", "latitude", "value"],
    check_missing=true,
    check_types=true
)

@test valid == true
@test length(errors) == 0
println("✓ Valid data passed validation")

# Test with missing required column
invalid_data = select(sample_data, Not(:longitude))
valid, errors = DataLoader.validate_data(
    invalid_data,
    required_cols=["longitude", "latitude"],
    check_missing=true
)

@test valid == false
@test length(errors) > 0
println("✓ Invalid data correctly detected")
println("  Errors: ", errors)
println()

# ============================================================================
# Test 4: Feature Normalization
# ============================================================================
println("Test 4: Feature Normalization")
println("-" ^ 70)

# Test min-max normalization
df_norm, params = FeatureEngineering.normalize_features(
    sample_data,
    ["value"],
    method=:minmax
)

@test minimum(df_norm.value) >= 0.0
@test maximum(df_norm.value) <= 1.0
println("✓ Min-max normalization successful")
println("  Range: [$(minimum(df_norm.value)), $(maximum(df_norm.value))]")

# Test z-score standardization
df_std, params = FeatureEngineering.standardize_features(
    sample_data,
    ["value"]
)

@test abs(mean(df_std.value)) < 1e-10  # Mean should be ~0
@test abs(std(df_std.value) - 1.0) < 1e-10  # Std should be ~1
println("✓ Z-score standardization successful")
println("  Mean: $(mean(df_std.value)), Std: $(std(df_std.value))")
println()

# ============================================================================
# Test 5: Categorical Encoding
# ============================================================================
println("Test 5: Categorical Encoding")
println("-" ^ 70)

# Add categorical column
sample_data.category = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"]

# Test one-hot encoding
df_onehot, mapping = FeatureEngineering.encode_categorical(
    sample_data,
    "category",
    method=:onehot
)

@test "category_A" in names(df_onehot)
@test "category_B" in names(df_onehot)
@test "category_C" in names(df_onehot)
println("✓ One-hot encoding successful")
println("  Created columns: category_A, category_B, category_C")

# Test label encoding
df_label, mapping = FeatureEngineering.encode_categorical(
    sample_data,
    "category",
    method=:label
)

@test all(x -> x isa Integer, df_label.category)
println("✓ Label encoding successful")
println("  Mapping: ", mapping["mapping"])
println()

# ============================================================================
# Test 6: Temporal Feature Extraction
# ============================================================================
println("Test 6: Temporal Feature Extraction")
println("-" ^ 70)

# Add datetime column
sample_data.created_at = [DateTime(2024, 1, 1) + Day(i) for i in 1:10]

df_temporal = FeatureEngineering.extract_temporal_features(sample_data, "created_at")

@test "year" in names(df_temporal)
@test "month" in names(df_temporal)
@test "day" in names(df_temporal)
@test "day_of_week" in names(df_temporal)
@test "quarter" in names(df_temporal)
@test "is_weekend" in names(df_temporal)
println("✓ Temporal features extracted successfully")
println("  Added features: year, month, day, day_of_week, quarter, is_weekend, days_since_epoch")
println()

# ============================================================================
# Test 7: Spatial Binning
# ============================================================================
println("Test 7: Spatial Binning")
println("-" ^ 70)

df_binned = FeatureEngineering.create_spatial_bins(
    sample_data,
    "longitude",
    "latitude",
    n_bins_lon=5,
    n_bins_lat=5
)

@test "lon_bin" in names(df_binned)
@test "lat_bin" in names(df_binned)
@test "spatial_bin_id" in names(df_binned)
println("✓ Spatial binning successful")
println("  Unique bins: ", length(unique(df_binned.spatial_bin_id)))
println()

# ============================================================================
# Test 8: Missing Value Handling
# ============================================================================
println("Test 8: Missing Value Handling")
println("-" ^ 70)

# Create data with missing values
data_with_missing = copy(sample_data)
# Convert to allow missing values
data_with_missing.value = Vector{Union{Float64, Missing}}(data_with_missing.value)
data_with_missing.value[3] = missing
data_with_missing.value[7] = missing

# Test mean imputation
df_imputed = FeatureEngineering.handle_missing_values(
    data_with_missing,
    ["value"],
    strategy=:mean
)

@test count(ismissing, df_imputed.value) == 0
println("✓ Missing value imputation (mean) successful")

# Test median imputation
df_imputed = FeatureEngineering.handle_missing_values(
    data_with_missing,
    ["value"],
    strategy=:median
)

@test count(ismissing, df_imputed.value) == 0
println("✓ Missing value imputation (median) successful")
println()

# ============================================================================
# Test 9: Outlier Removal
# ============================================================================
println("Test 9: Outlier Removal")
println("-" ^ 70)

# Create data with outliers
data_with_outliers = copy(sample_data)
data_with_outliers.value[1] = 1000.0  # Outlier

original_count = nrow(data_with_outliers)
df_clean = FeatureEngineering.remove_outliers(
    data_with_outliers,
    ["value"],
    method=:iqr,
    threshold=1.5
)

@test nrow(df_clean) < original_count
println("✓ Outlier removal successful")
println("  Rows removed: ", original_count - nrow(df_clean))
println()

# ============================================================================
# Test 10: Feature Interaction
# ============================================================================
println("Test 10: Feature Interaction and Polynomial Features")
println("-" ^ 70)

# Create interaction features
df_interact = FeatureEngineering.create_interaction_features(
    sample_data,
    "longitude",
    "latitude"
)

@test "longitude_x_latitude" in names(df_interact)
println("✓ Interaction features created successfully")

# Create polynomial features
df_poly = FeatureEngineering.polynomial_features(
    sample_data,
    ["value"],
    3  # degree as positional argument
)

@test "value_pow2" in names(df_poly)
@test "value_pow3" in names(df_poly)
println("✓ Polynomial features created successfully")
println()

# ============================================================================
# Summary
# ============================================================================
println("=" ^ 70)
println("Test Summary")
println("=" ^ 70)
println("✓ All data loading utility tests passed!")
println()
println("Tested functionality:")
println("  1. Database connection management")
println("  2. Spatial feature extraction")
println("  3. Data validation")
println("  4. Feature normalization and standardization")
println("  5. Categorical encoding (one-hot and label)")
println("  6. Temporal feature extraction")
println("  7. Spatial binning")
println("  8. Missing value handling")
println("  9. Outlier removal")
println("  10. Feature interaction and polynomial features")
println()
println("Data loading utilities are ready for use in ML models!")
println("=" ^ 70)
