# Test Anomaly Detection with Isolation Forest Only
# Simplified version for quick validation

using Pkg
Pkg.activate(".")

using DataFrames
using Random
using Statistics

println("=" ^ 80)
println("Anomaly Detection Test - Isolation Forest Implementation")
println("=" ^ 80)

Random.seed!(42)

# ============================================================================
# Simplified Isolation Forest Implementation
# ============================================================================

"""Build a simple isolation tree"""
function build_tree(X::Matrix{Float64}, depth::Int, max_depth::Int)
    n_samples, n_features = size(X)
    
    # Stop conditions
    if depth >= max_depth || n_samples <= 1
        return Dict("type" => "leaf", "size" => n_samples, "depth" => depth)
    end
    
    # Random split
    split_feature = rand(1:n_features)
    feature_values = X[:, split_feature]
    
    if length(unique(feature_values)) <= 1
        return Dict("type" => "leaf", "size" => n_samples, "depth" => depth)
    end
    
    split_value = minimum(feature_values) + rand() * (maximum(feature_values) - minimum(feature_values))
    
    # Split data
    left_mask = feature_values .< split_value
    right_mask = .!left_mask
    
    if sum(left_mask) == 0 || sum(right_mask) == 0
        return Dict("type" => "leaf", "size" => n_samples, "depth" => depth)
    end
    
    return Dict(
        "type" => "node",
        "feature" => split_feature,
        "value" => split_value,
        "left" => build_tree(X[left_mask, :], depth + 1, max_depth),
        "right" => build_tree(X[right_mask, :], depth + 1, max_depth)
    )
end

"""Compute path length for a sample"""
function get_path_length(tree::Dict, x::Vector{Float64}, depth::Int=0)
    if tree["type"] == "leaf"
        return Float64(depth)
    end
    
    feature = tree["feature"]
    value = tree["value"]
    
    if x[feature] < value
        return get_path_length(tree["left"], x, depth + 1)
    else
        return get_path_length(tree["right"], x, depth + 1)
    end
end

# ============================================================================
# Test
# ============================================================================

println("\n1. Generating training data (normal transactions)...")
n_train = 500
X_train = DataFrame(
    amount = 1000 .+ 200 .* randn(n_train),
    time_of_day = rand(9:17, n_train) .+ rand(n_train),
    category = Float64.(rand(1:5, n_train)),
    location = Float64.(rand(1:10, n_train))
)
println("   ✓ Generated $(n_train) normal transactions")

println("\n2. Training Isolation Forest...")
n_trees = 100
max_depth = ceil(Int, log2(256))
trees = []

X_matrix = Matrix{Float64}(X_train)

for i in 1:n_trees
    # Sample for this tree
    sample_size = min(256, n_train)
    sample_indices = rand(1:n_train, sample_size)
    X_sample = X_matrix[sample_indices, :]
    
    tree = build_tree(X_sample, 0, max_depth)
    push!(trees, tree)
end

println("   ✓ Trained $(n_trees) trees")

println("\n3. Testing on normal transactions...")
X_test_normal = DataFrame(
    amount = 1000 .+ 200 .* randn(50),
    time_of_day = rand(9:17, 50) .+ rand(50),
    category = Float64.(rand(1:5, 50)),
    location = Float64.(rand(1:10, 50))
)

X_test_matrix = Matrix{Float64}(X_test_normal)
scores_normal = zeros(50)

for i in 1:50
    x = X_test_matrix[i, :]
    avg_path = mean([get_path_length(tree, x) for tree in trees])
    
    # Anomaly score: shorter paths = higher score
    # Normalize: typical path length is around log2(256) ≈ 8
    # Score in [0, 1] where 1 = very anomalous
    normalized_score = 1.0 - (avg_path / (max_depth + 1))
    scores_normal[i] = max(0.0, normalized_score)
end

avg_normal = mean(scores_normal)
println("   Average score: $(round(avg_normal, digits=3))")
println("   Score range: [$(round(minimum(scores_normal), digits=3)), $(round(maximum(scores_normal), digits=3))]")

println("\n4. Testing on anomalous transactions...")
X_test_anomalous = DataFrame(
    amount = 5000 .+ 1000 .* randn(50),  # Much higher amounts
    time_of_day = rand([2, 3, 4, 22, 23], 50) .+ rand(50),  # Unusual times
    category = Float64.(rand(1:5, 50)),
    location = Float64.(rand(1:10, 50))
)

X_test_anom_matrix = Matrix{Float64}(X_test_anomalous)
scores_anomalous = zeros(50)

for i in 1:50
    x = X_test_anom_matrix[i, :]
    avg_path = mean([get_path_length(tree, x) for tree in trees])
    
    normalized_score = 1.0 - (avg_path / (max_depth + 1))
    scores_anomalous[i] = max(0.0, normalized_score)
end

avg_anomalous = mean(scores_anomalous)
println("   Average score: $(round(avg_anomalous, digits=3))")
println("   Score range: [$(round(minimum(scores_anomalous), digits=3)), $(round(maximum(scores_anomalous), digits=3))]")

println("\n5. Results:")
println("   Normal transactions avg score: $(round(avg_normal, digits=3))")
println("   Anomalous transactions avg score: $(round(avg_anomalous, digits=3))")

if avg_anomalous > avg_normal
    diff = round((avg_anomalous - avg_normal) / avg_normal * 100, digits=1)
    println("\n   ✓ SUCCESS: Anomalous scores are $(diff)% higher than normal!")
else
    println("\n   ⚠ WARNING: Anomalous scores not significantly higher")
end

# Classification test
threshold = 0.5
normal_flagged = count(s -> s >= threshold, scores_normal)
anomalous_detected = count(s -> s >= threshold, scores_anomalous)

println("\n6. Classification (threshold = $(threshold)):")
println("   Normal flagged as anomalies: $(normal_flagged)/50 ($(round(normal_flagged/50*100, digits=1))%)")
println("   Anomalies detected: $(anomalous_detected)/50 ($(round(anomalous_detected/50*100, digits=1))%)")

println("\n" * "=" ^ 80)
println("Isolation Forest Test Complete!")
println("=" ^ 80)
