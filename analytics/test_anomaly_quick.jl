# Quick Test for Anomaly Detection Module
# Minimal test to verify basic functionality

using Pkg
Pkg.activate(".")

using DataFrames
using Random
using Statistics

# Import module
include("src/models/anomaly_detector.jl")
using .AnomalyDetector

println("=" ^ 80)
println("Quick Anomaly Detection Test")
println("=" ^ 80)

Random.seed!(42)

# Generate simple test data
println("\n1. Generating test data...")
n_normal = 200
n_anomalous = 20

# Normal transactions
X_train = DataFrame(
    amount = 1000 .+ 200 .* randn(n_normal),
    time_of_day = rand(9:17, n_normal) .+ rand(n_normal),
    category = Float64.(rand(1:5, n_normal)),
    location = Float64.(rand(1:10, n_normal))
)

println("   ✓ Generated $(n_normal) normal transactions for training")

# Test data (mix of normal and anomalous)
X_test_normal = DataFrame(
    amount = 1000 .+ 200 .* randn(50),
    time_of_day = rand(9:17, 50) .+ rand(50),
    category = Float64.(rand(1:5, 50)),
    location = Float64.(rand(1:10, 50))
)

X_test_anomalous = DataFrame(
    amount = 5000 .+ 1000 .* randn(n_anomalous),
    time_of_day = rand([2, 3, 4, 22, 23], n_anomalous) .+ rand(n_anomalous),
    category = Float64.(rand(1:5, n_anomalous)),
    location = Float64.(rand(1:10, n_anomalous))
)

println("   ✓ Generated 50 normal + $(n_anomalous) anomalous test transactions")

# Train model
println("\n2. Training ensemble anomaly detection model...")
println("   (This may take 1-2 minutes...)")

model = train_anomaly_model(
    X_train, 
    contamination=0.1, 
    autoencoder_epochs=30,  # Reduced for quick test
    threshold=0.5
)

println("   ✓ Model trained successfully")
println("   - Isolation Forest: trained")
println("   - Autoencoder: trained")
println("   - Features: $(length(model.feature_names))")

# Test on normal transactions
println("\n3. Testing on normal transactions...")
results_normal = detect_anomalies(model, X_test_normal)

normal_anomaly_count = count(r -> r.is_anomaly, results_normal)
normal_anomaly_rate = normal_anomaly_count / length(results_normal) * 100

println("   ✓ Analyzed $(length(results_normal)) normal transactions")
println("   - Flagged as anomalies: $(normal_anomaly_count) ($(round(normal_anomaly_rate, digits=1))%)")
println("   - Average score: $(round(mean(r.anomaly_score for r in results_normal), digits=3))")

# Test on anomalous transactions
println("\n4. Testing on anomalous transactions...")
results_anomalous = detect_anomalies(model, X_test_anomalous)

anomalous_detected = count(r -> r.is_anomaly, results_anomalous)
detection_rate = anomalous_detected / length(results_anomalous) * 100

println("   ✓ Analyzed $(length(results_anomalous)) anomalous transactions")
println("   - Detected as anomalies: $(anomalous_detected) ($(round(detection_rate, digits=1))%)")
println("   - Average score: $(round(mean(r.anomaly_score for r in results_anomalous), digits=3))")

# Severity breakdown
println("\n5. Severity breakdown for detected anomalies:")
severity_counts = Dict("low" => 0, "medium" => 0, "high" => 0, "critical" => 0)
for r in results_anomalous
    if r.is_anomaly
        severity_counts[r.severity] += 1
    end
end

for (severity, count) in sort(collect(severity_counts), by=x->x[2], rev=true)
    if count > 0
        println("   - $(severity): $(count)")
    end
end

# Single transaction test
println("\n6. Testing single transaction scoring...")
single_transaction = DataFrame(
    amount = [8000.0],
    time_of_day = [2.5],
    category = [1.0],
    location = [5.0]
)

score = compute_anomaly_score(model, single_transaction)
println("   ✓ Single transaction anomaly score: $(round(score, digits=3))")

# Summary
println("\n" * "=" ^ 80)
println("Quick Test Complete!")
println("=" ^ 80)
println("\nResults Summary:")
println("  ✓ Model training: SUCCESS")
println("  ✓ Normal transaction detection: $(round(100 - normal_anomaly_rate, digits=1))% correctly classified")
println("  ✓ Anomalous transaction detection: $(round(detection_rate, digits=1))% detected")
println("  ✓ Single transaction scoring: WORKING")

avg_normal = mean(r.anomaly_score for r in results_normal)
avg_anomalous = mean(r.anomaly_score for r in results_anomalous)

if avg_anomalous > avg_normal
    println("\n  ✓ VALIDATION: Anomalous transactions have higher scores ($(round(avg_anomalous, digits=3)) vs $(round(avg_normal, digits=3)))")
else
    println("\n  ⚠ WARNING: Anomalous scores not significantly higher than normal")
end

println("\n" * "=" ^ 80)
