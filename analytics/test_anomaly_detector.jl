# Test Anomaly Detection Module
# Tests for Tasks 14.1-14.3 and 14.15

using Test
using DataFrames
using Random
using Statistics

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "src", "models"))
push!(LOAD_PATH, joinpath(@__DIR__, "src", "explainability"))

# Import modules
include("src/models/anomaly_detector.jl")
include("src/explainability/shap_engine.jl")

using .AnomalyDetector
using .SHAPEngine

println("=" ^ 80)
println("Testing Anomaly Detection Module")
println("=" ^ 80)

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# Test Data Generation
# ============================================================================

"""Generate synthetic normal transaction data"""
function generate_normal_transactions(n::Int=1000)
    # Normal transactions: amount around 1000, time during business hours
    amounts = 1000 .+ 200 .* randn(n)
    times = rand(9:17, n) .+ rand(n)  # Business hours 9-17
    categories = rand(1:5, n)
    locations = rand(1:10, n)
    
    return DataFrame(
        amount = amounts,
        time_of_day = times,
        category = Float64.(categories),
        location = Float64.(locations)
    )
end

"""Generate synthetic anomalous transactions"""
function generate_anomalous_transactions(n::Int=50)
    # Anomalous transactions: very high amounts, unusual times
    amounts = 5000 .+ 1000 .* randn(n)  # Much higher amounts
    times = rand([2, 3, 4, 22, 23], n) .+ rand(n)  # Late night/early morning
    categories = rand(1:5, n)
    locations = rand(1:10, n)
    
    return DataFrame(
        amount = amounts,
        time_of_day = times,
        category = Float64.(categories),
        location = Float64.(locations)
    )
end

# ============================================================================
# Test 1: Isolation Forest Training
# ============================================================================

@testset "Task 14.1: Isolation Forest" begin
    println("\n" * "=" ^ 80)
    println("Test 1: Isolation Forest Training and Scoring")
    println("=" ^ 80)
    
    # Generate training data
    X_train = generate_normal_transactions(500)
    
    # Train Isolation Forest
    println("\nTraining Isolation Forest...")
    if_model = AnomalyDetector.train_isolation_forest(X_train, contamination=0.1, n_trees=50)
    
    @test if_model !== nothing
    println("✓ Isolation Forest trained successfully")
    
    # Test scoring
    X_test = generate_normal_transactions(100)
    scores = AnomalyDetector.compute_isolation_forest_score(if_model, X_test)
    
    @test length(scores) == 100
    @test all(isfinite.(scores))
    println("✓ Isolation Forest scoring works")
    println("  Score range: [$(minimum(scores)), $(maximum(scores))]")
end

# ============================================================================
# Test 2: Autoencoder Training
# ============================================================================

@testset "Task 14.2: Autoencoder" begin
    println("\n" * "=" ^ 80)
    println("Test 2: Autoencoder Training and Reconstruction")
    println("=" ^ 80)
    
    # Generate training data
    X_train = generate_normal_transactions(500)
    
    # Train Autoencoder
    println("\nTraining Autoencoder...")
    autoencoder, encoder = AnomalyDetector.train_autoencoder(X_train, epochs=50, learning_rate=0.01)
    
    @test autoencoder !== nothing
    @test encoder !== nothing
    println("✓ Autoencoder trained successfully")
    
    # Test reconstruction
    X_test = generate_normal_transactions(100)
    errors = AnomalyDetector.compute_reconstruction_error(autoencoder, X_test)
    
    @test length(errors) == 100
    @test all(isfinite.(errors))
    @test all(errors .>= 0)
    println("✓ Autoencoder reconstruction works")
    println("  Reconstruction error range: [$(minimum(errors)), $(maximum(errors))]")
end

# ============================================================================
# Test 3: Ensemble Anomaly Detection
# ============================================================================

@testset "Task 14.3: Ensemble Detection" begin
    println("\n" * "=" ^ 80)
    println("Test 3: Ensemble Anomaly Detection")
    println("=" ^ 80)
    
    # Generate training data (normal transactions only)
    X_train = generate_normal_transactions(500)
    
    # Train ensemble model
    println("\nTraining ensemble anomaly detection model...")
    model = train_anomaly_model(X_train, contamination=0.1, autoencoder_epochs=50, threshold=0.5)
    
    @test model !== nothing
    @test model.isolation_forest !== nothing
    @test model.autoencoder !== nothing
    @test model.threshold == 0.5
    @test length(model.feature_names) == 4
    println("✓ Ensemble model trained successfully")
    
    # Test on normal transactions
    println("\nTesting on normal transactions...")
    X_normal = generate_normal_transactions(100)
    results_normal = detect_anomalies(model, X_normal)
    
    @test length(results_normal) == 100
    @test all(r -> 0 <= r.anomaly_score <= 1, results_normal)
    
    normal_anomaly_rate = count(r -> r.is_anomaly, results_normal) / length(results_normal)
    println("  Normal transactions flagged as anomalies: $(round(normal_anomaly_rate * 100, digits=1))%")
    
    # Test on anomalous transactions
    println("\nTesting on anomalous transactions...")
    X_anomalous = generate_anomalous_transactions(50)
    results_anomalous = detect_anomalies(model, X_anomalous)
    
    @test length(results_anomalous) == 50
    @test all(r -> 0 <= r.anomaly_score <= 1, results_anomalous)
    
    anomalous_detection_rate = count(r -> r.is_anomaly, results_anomalous) / length(results_anomalous)
    println("  Anomalous transactions detected: $(round(anomalous_detection_rate * 100, digits=1))%")
    
    # Verify anomalous transactions have higher scores
    avg_normal_score = mean(r.anomaly_score for r in results_normal)
    avg_anomalous_score = mean(r.anomaly_score for r in results_anomalous)
    
    println("\n  Average scores:")
    println("    Normal: $(round(avg_normal_score, digits=3))")
    println("    Anomalous: $(round(avg_anomalous_score, digits=3))")
    
    @test avg_anomalous_score > avg_normal_score
    println("✓ Anomalous transactions have higher scores than normal")
end

# ============================================================================
# Test 4: Score Normalization
# ============================================================================

@testset "Score Normalization" begin
    println("\n" * "=" ^ 80)
    println("Test 4: Score Normalization")
    println("=" ^ 80)
    
    # Test normalization
    raw_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = AnomalyDetector.normalize_scores(raw_scores)
    
    @test minimum(normalized) == 0.0
    @test maximum(normalized) == 1.0
    @test all(0 .<= normalized .<= 1)
    println("✓ Score normalization works correctly")
    println("  Raw: $(raw_scores)")
    println("  Normalized: $(normalized)")
end

# ============================================================================
# Test 5: Severity Classification
# ============================================================================

@testset "Severity Classification" begin
    println("\n" * "=" ^ 80)
    println("Test 5: Severity Classification")
    println("=" ^ 80)
    
    @test AnomalyDetector.classify_severity(0.2) == "low"
    @test AnomalyDetector.classify_severity(0.5) == "medium"
    @test AnomalyDetector.classify_severity(0.7) == "high"
    @test AnomalyDetector.classify_severity(0.9) == "critical"
    
    println("✓ Severity classification works correctly")
    println("  0.2 → low")
    println("  0.5 → medium")
    println("  0.7 → high")
    println("  0.9 → critical")
end

# ============================================================================
# Test 6: Deviation Metrics
# ============================================================================

@testset "Deviation Metrics" begin
    println("\n" * "=" ^ 80)
    println("Test 6: Deviation Metrics")
    println("=" ^ 80)
    
    reference = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Test normal value
    metrics_normal = AnomalyDetector.compute_deviation_metrics(3.0, reference)
    @test abs(metrics_normal["z_score"]) < 0.1
    @test 40 <= metrics_normal["percentile_rank"] <= 60
    
    # Test outlier value
    metrics_outlier = AnomalyDetector.compute_deviation_metrics(10.0, reference)
    @test metrics_outlier["z_score"] > 2.0
    @test metrics_outlier["percentile_rank"] == 100.0
    
    println("✓ Deviation metrics computed correctly")
    println("  Normal value (3.0): z=$(round(metrics_normal["z_score"], digits=2)), percentile=$(round(metrics_normal["percentile_rank"], digits=1))")
    println("  Outlier value (10.0): z=$(round(metrics_outlier["z_score"], digits=2)), percentile=$(round(metrics_outlier["percentile_rank"], digits=1))")
end

# ============================================================================
# Test 7: Single Transaction Scoring
# ============================================================================

@testset "Single Transaction Scoring" begin
    println("\n" * "=" ^ 80)
    println("Test 7: Single Transaction Scoring")
    println("=" ^ 80)
    
    # Train model
    X_train = generate_normal_transactions(500)
    model = train_anomaly_model(X_train, contamination=0.1, autoencoder_epochs=50)
    
    # Test single transaction
    transaction = DataFrame(
        amount = [10000.0],  # Very high amount
        time_of_day = [3.5],  # Late night
        category = [1.0],
        location = [5.0]
    )
    
    score = compute_anomaly_score(model, transaction)
    
    @test 0 <= score <= 1
    println("✓ Single transaction scoring works")
    println("  Anomaly score: $(round(score, digits=3))")
end

# ============================================================================
# Test 8: SHAP Integration
# ============================================================================

@testset "Task 14.15: SHAP Integration" begin
    println("\n" * "=" ^ 80)
    println("Test 8: SHAP Integration for Anomaly Explanations")
    println("=" ^ 80)
    
    # Train model
    X_train = generate_normal_transactions(500)
    model = train_anomaly_model(X_train, contamination=0.1, autoencoder_epochs=50)
    
    # Create a prediction function for SHAP
    function anomaly_predict_fn(data::DataFrame)
        results = detect_anomalies(model, data)
        return [r.anomaly_score for r in results]
    end
    
    # Test transaction
    X_test = generate_anomalous_transactions(1)
    
    println("\nComputing SHAP explanation for anomaly...")
    shap_result = SHAPEngine.compute_shap_local(
        anomaly_predict_fn,
        X_train,
        X_test,
        model_type="regression"
    )
    
    @test shap_result !== nothing
    @test haskey(shap_result, "shap_values")
    @test haskey(shap_result, "base_value")
    @test haskey(shap_result, "prediction")
    
    println("✓ SHAP computation successful")
    
    # Generate explanation
    explanation = SHAPEngine.format_explanation(shap_result, names(X_test))
    
    @test haskey(explanation, "feature_importance")
    @test haskey(explanation, "summary")
    
    println("\n  SHAP Explanation:")
    println("  Base value: $(round(explanation["base_value"], digits=3))")
    println("  Prediction: $(round(explanation["prediction"], digits=3))")
    println("\n  Top contributing features:")
    for (i, feat) in enumerate(explanation["feature_importance"][1:min(3, end)])
        println("    $(i). $(feat["feature"]): $(round(feat["shap_value"], digits=3))")
    end
    
    println("\n  Summary: $(explanation["summary"])")
    
    println("\n✓ SHAP integration with anomaly detection complete")
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 80)
println("All Anomaly Detection Tests Passed!")
println("=" ^ 80)
println("\nImplemented Features:")
println("  ✓ Task 14.1: Isolation Forest anomaly detector")
println("  ✓ Task 14.2: Autoencoder anomaly detector")
println("  ✓ Task 14.3: Ensemble anomaly detection")
println("  ✓ Task 14.15: SHAP integration for explainability")
println("\nRequirements Satisfied:")
println("  ✓ Requirement 5.1: Ensemble anomaly detection")
println("  ✓ Requirement 5.2: Normalized scores and severity classification")
println("  ✓ Requirement 5.3: SHAP explanations for anomalies")
println("  ✓ Requirement 5.4: Deviation metrics")
println("\n" * "=" ^ 80)
