# Direct Test for ROI Predictor Module
# Tests the ROI predictor module directly without loading full AnalyticsEngine

using Pkg
Pkg.activate(@__DIR__)

println("Loading dependencies...")
using DataFrames
using Random
using MLJ
using XGBoost
using Flux
using Statistics
using StatsBase

println("Loading ROI Predictor module...")
include("src/models/roi_predictor.jl")
using .ROIPredictor

println("=" ^ 80)
println("ROI Predictor Direct Test")
println("=" ^ 80)

Random.seed!(42)

println("\n1. Generating training data...")
n_samples = 100
training_data = DataFrame(
    latitude = 20.0 .+ rand(n_samples) .* 15.0,
    longitude = 70.0 .+ rand(n_samples) .* 25.0,
    sector_agriculture = rand([0.0, 1.0], n_samples),
    investment_amount = 50000.0 .+ rand(n_samples) .* 950000.0,
    timeframe_years = rand(1:10, n_samples),
    infrastructure_score = rand(n_samples),
    roi = 0.05 .+ rand(n_samples) .* 0.30
)

println("✓ Generated $(nrow(training_data)) samples")

println("\n2. Training ensemble model...")
try
    model = ROIPredictor.train_ensemble_model(training_data, target_col="roi")
    println("✓ Model trained")
    
    println("\n3. Making prediction...")
    test_instance = DataFrame(
        latitude = [28.6],
        longitude = [77.2],
        sector_agriculture = [1.0],
        investment_amount = [500000.0],
        timeframe_years = [5],
        infrastructure_score = [0.8]
    )
    
    predictions = ROIPredictor.predict_roi(model, test_instance)
    pred = predictions[1]
    
    println("✓ Prediction: $(round(pred.predicted_roi * 100, digits=2))%")
    println("  CI: [$(round(pred.confidence_lower * 100, digits=2))%, $(round(pred.confidence_upper * 100, digits=2))%]")
    println("  Variance: $(round(pred.variance, digits=6))")
    
    println("\n✓ All tests passed!")
    
catch e
    println("\n✗ Error:")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end
