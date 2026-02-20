# Simplified Anomaly Detection Test
# Just test the Isolation Forest part first

using Pkg
Pkg.activate(".")

using DataFrames
using Random
using Statistics

# Import module
include("src/models/anomaly_detector.jl")
using .AnomalyDetector

println("=" ^ 80)
println("Simple Anomaly Detection Test (Isolation Forest Only)")
println("=" ^ 80)

Random.seed!(42)

# Generate simple test data
println("\n1. Generating test data...")
X_train = DataFrame(
    amount = 1000 .+ 200 .* randn(100),
    time_of_day = rand(9:17, 100) .+ rand(100)
)

println("   ✓ Generated 100 training samples")

# Train Isolation Forest only
println("\n2. Training Isolation Forest...")
if_model = AnomalyDetector.train_isolation_forest(X_train, contamination=0.1, n_trees=50)

println("   ✓ Isolation Forest trained")
println("   - Trees: $(if_model["n_trees"])")
println("   - Features: $(if_model["n_features"])")

# Test scoring
println("\n3. Testing anomaly scoring...")

# Normal transaction
X_normal = DataFrame(
    amount = [1000.0],
    time_of_day = [12.0]
)

score_normal = AnomalyDetector.compute_isolation_forest_score(if_model, X_normal)[1]
println("   Normal transaction score: $(round(score_normal, digits=3))")

# Anomalous transaction
X_anomalous = DataFrame(
    amount = [10000.0],
    time_of_day = [3.0]
)

score_anomalous = AnomalyDetector.compute_isolation_forest_score(if_model, X_anomalous)[1]
println("   Anomalous transaction score: $(round(score_anomalous, digits=3))")

if score_anomalous > score_normal
    println("\n   ✓ SUCCESS: Anomalous transaction has higher score!")
else
    println("\n   ⚠ Anomalous score not higher (may need more training data)")
end

println("\n" * "=" ^ 80)
println("Isolation Forest Test Complete!")
println("=" ^ 80)
