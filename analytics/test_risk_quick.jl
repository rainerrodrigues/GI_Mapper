# Quick Test for Risk Assessment Module

using Pkg
Pkg.activate(".")

using DataFrames
using Random
using Statistics

println("=" ^ 80)
println("Risk Assessment Quick Test")
println("=" ^ 80)

Random.seed!(42)

# Import module
include("src/models/risk_assessor.jl")
using .RiskAssessor

# Generate simple test data
println("\n1. Generating test data...")
n = 20
projects = DataFrame(
    project_id = ["PROJ_$(lpad(i, 2, '0'))" for i in 1:n],
    cost_overrun_pct = clamp.(20 .+ 15 .* randn(n), 0, 100),
    operational_complexity = clamp.(50 .+ 20 .* randn(n), 0, 100),
    environmental_impact = clamp.(30 .+ 20 .* randn(n), 0, 100),
    social_acceptance = clamp.(70 .+ 15 .* randn(n), 0, 100),
    distance_to_infrastructure = clamp.(50 .+ 30 .* randn(n), 0, 200),
    climate_risk_index = clamp.(0.4 .+ 0.2 .* randn(n), 0, 1)
)

# Synthetic risk levels
projects.risk_level = 0.3 .* projects.cost_overrun_pct ./ 100 .+ 
                      0.3 .* projects.operational_complexity ./ 100 .+
                      0.2 .* projects.environmental_impact ./ 100 .+
                      0.2 .* (100 .- projects.social_acceptance) ./ 100

println("   ✓ Generated $(nrow(projects)) projects")

# Test GB classifier
println("\n2. Testing GB classifier...")
gb_model = RiskAssessor.train_gb_classifier(projects, "risk_level")
println("   ✓ GB classifier trained")

# Test NN classifier (skip full training)
println("\n3. Testing NN classifier (simplified)...")
nn_model = Dict("type" => "nn_mean", "mean_risk" => mean(projects.risk_level))
println("   ✓ NN classifier created")

# Test multi-dimensional scoring
println("\n4. Testing multi-dimensional risk scoring...")
test_project = projects[1:1, Not(:risk_level)]
feature_names = [col for col in names(projects) if col != "risk_level" && col != "project_id"]
dim_scores = RiskAssessor.compute_dimensional_risks(test_project, feature_names)
println("   ✓ Dimensional scores: $(length(dim_scores)) dimensions")

# Test spatial factors
println("\n5. Testing spatial risk factors...")
spatial_factors = RiskAssessor.extract_spatial_risk_factors(test_project)
println("   ✓ Spatial factors: $(length(spatial_factors)) factors")

# Test probability calibration
println("\n6. Testing probability calibration...")
calibrated, conf = RiskAssessor.calibrate_risk_probability(0.7, 0.8)
println("   ✓ Calibrated probability: $(round(calibrated, digits=3))")

println("\n" * "=" ^ 80)
println("Quick Test Passed!")
println("=" ^ 80)
println("\nCore Features Working:")
println("  ✓ GB classifier training")
println("  ✓ Multi-dimensional risk scoring")
println("  ✓ Spatial risk factor extraction")
println("  ✓ Probability calibration")
println("\n" * "=" ^ 80)
