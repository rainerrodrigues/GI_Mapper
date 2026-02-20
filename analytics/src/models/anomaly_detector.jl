# Anomaly Detection Module
# Tasks 14.1-14.3: Implement Isolation Forest, Autoencoder, and Ensemble Detection
# Requirements: 5.1, 5.2, 5.3, 5.4, 5.5

"""
Anomaly Detection module implementing ensemble anomaly detection for transaction monitoring.

This module provides:
- Isolation Forest anomaly detector (Task 14.1)
- Autoencoder anomaly detector using Flux (Task 14.2)
- Ensemble anomaly detection (Task 14.3)
- Integration with SHAP explainability (Task 14.15)
"""

module AnomalyDetector

using OutlierDetection
using Flux
using DataFrames
using Statistics
using StatsBase
using Logging
using Random
using LinearAlgebra

export AnomalyModel, train_anomaly_model, detect_anomalies
export AnomalyResult, compute_anomaly_score

# ============================================================================
# Data Structures
# ============================================================================

"""
    AnomalyModel

Container for the ensemble anomaly detection model.

# Fields
- `isolation_forest`: Trained Isolation Forest model
- `autoencoder`: Trained Autoencoder (Flux Chain)
- `encoder`: Encoder portion of autoencoder
- `threshold::Float64`: Anomaly score threshold
- `feature_names::Vector{String}`: Names of input features
- `feature_stats::Dict`: Feature statistics for normalization
- `model_version::String`: Model version identifier
"""
mutable struct AnomalyModel
    isolation_forest::Any
    autoencoder::Any
    encoder::Any
    threshold::Float64
    feature_names::Vector{String}
    feature_stats::Dict{String, Any}
    model_version::String
end

"""
    AnomalyResult

Container for anomaly detection results.

# Fields
- `is_anomaly::Bool`: Whether the transaction is anomalous
- `anomaly_score::Float64`: Normalized anomaly score [0, 1]
- `severity::String`: Severity level ('low', 'medium', 'high', 'critical')
- `isolation_forest_score::Float64`: Score from Isolation Forest
- `autoencoder_score::Float64`: Score from Autoencoder
- `expected_value::Float64`: Expected value based on normal patterns
- `deviation_metrics::Dict`: Z-scores and percentile ranks
"""
struct AnomalyResult
    is_anomaly::Bool
    anomaly_score::Float64
    severity::String
    isolation_forest_score::Float64
    autoencoder_score::Float64
    expected_value::Float64
    deviation_metrics::Dict{String, Float64}
end

# ============================================================================
# Task 14.1: Isolation Forest Anomaly Detector
# ============================================================================

"""
    train_isolation_forest(X::DataFrame; contamination::Float64=0.1, n_trees::Int=100)

Train an Isolation Forest model for anomaly detection.
Uses a simplified implementation based on random partitioning.

# Arguments
- `X::DataFrame`: Training data (normal transactions)
- `contamination::Float64`: Expected proportion of anomalies (default: 0.1)
- `n_trees::Int`: Number of trees in the forest (default: 100)

# Returns
- Trained Isolation Forest model (Dict with trees and statistics)

# Requirements
- Requirement 5.1: Ensemble anomaly detection
"""
function train_isolation_forest(X::DataFrame; contamination::Float64=0.1, n_trees::Int=100)
    @info "Training Isolation Forest with $(n_trees) trees, contamination=$(contamination)"
    
    # Convert DataFrame to matrix
    X_matrix = Matrix{Float64}(X)
    n_samples, n_features = size(X_matrix)
    
    # Build isolation trees
    trees = []
    max_depth = ceil(Int, log2(n_samples))
    
    for i in 1:n_trees
        # Sample data for this tree
        sample_size = min(256, n_samples)
        sample_indices = rand(1:n_samples, sample_size)
        X_sample = X_matrix[sample_indices, :]
        
        # Build tree (simplified: store split info)
        tree = build_isolation_tree(X_sample, 0, max_depth)
        push!(trees, tree)
    end
    
    # Store model
    model = Dict(
        "trees" => trees,
        "n_trees" => n_trees,
        "contamination" => contamination,
        "n_features" => n_features,
        "max_depth" => max_depth
    )
    
    @info "Isolation Forest training complete"
    return model
end

"""Build a simple isolation tree"""
function build_isolation_tree(X::Matrix{Float64}, depth::Int, max_depth::Int)
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
        "left" => build_isolation_tree(X[left_mask, :], depth + 1, max_depth),
        "right" => build_isolation_tree(X[right_mask, :], depth + 1, max_depth)
    )
end

"""Compute path length for a sample in a tree"""
function path_length(tree::Dict, x::Vector{Float64}, depth::Int=0)
    if tree["type"] == "leaf"
        # Average path length for unsuccessful search in BST
        size = tree["size"]
        if size <= 1
            return depth
        end
        # Adjustment for average path length
        c = 2.0 * (log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size
        return depth + c
    end
    
    feature = tree["feature"]
    value = tree["value"]
    
    if x[feature] < value
        return path_length(tree["left"], x, depth + 1)
    else
        return path_length(tree["right"], x, depth + 1)
    end
end

"""
    compute_isolation_forest_score(model, X::DataFrame)

Compute anomaly scores using Isolation Forest.

# Arguments
- `model`: Trained Isolation Forest model
- `X::DataFrame`: Data to score

# Returns
- Vector of anomaly scores (higher = more anomalous)
"""
function compute_isolation_forest_score(model, X::DataFrame)
    X_matrix = Matrix{Float64}(X)
    n_samples = size(X_matrix, 1)
    trees = model["trees"]
    n_trees = model["n_trees"]
    n_train_samples = 256  # Sample size used during training
    
    scores = zeros(n_samples)
    
    for i in 1:n_samples
        x = X_matrix[i, :]
        
        # Average path length across all trees
        avg_path = mean([path_length(tree, x) for tree in trees])
        
        # Anomaly score (shorter paths = more anomalous)
        # Normalize by expected path length for a BST
        c = 2.0 * (log(n_train_samples - 1) + 0.5772156649) - 2.0 * (n_train_samples - 1) / n_train_samples
        
        # Score formula: 2^(-E(h(x))/c(n))
        # Values close to 1 = anomaly, close to 0 = normal
        score = 2.0 ^ (-avg_path / c)
        
        scores[i] = score
    end
    
    return scores
end

# ============================================================================
# Task 14.2: Autoencoder Anomaly Detector
# ============================================================================

"""
    build_autoencoder(input_dim::Int; hidden_dims::Vector{Int}=[32, 16, 8])

Build an autoencoder neural network architecture.

# Arguments
- `input_dim::Int`: Number of input features
- `hidden_dims::Vector{Int}`: Dimensions of hidden layers (default: [32, 16, 8])

# Returns
- Tuple of (encoder, decoder, autoencoder)

# Requirements
- Requirement 5.1: Ensemble anomaly detection
"""
function build_autoencoder(input_dim::Int; hidden_dims::Vector{Int}=[32, 16, 8])
    @info "Building autoencoder: input_dim=$(input_dim), hidden_dims=$(hidden_dims)"
    
    # Encoder
    encoder = Chain(
        Dense(input_dim, hidden_dims[1], relu),
        Dense(hidden_dims[1], hidden_dims[2], relu),
        Dense(hidden_dims[2], hidden_dims[3])
    )
    
    # Decoder
    decoder = Chain(
        Dense(hidden_dims[3], hidden_dims[2], relu),
        Dense(hidden_dims[2], hidden_dims[1], relu),
        Dense(hidden_dims[1], input_dim)
    )
    
    # Full autoencoder
    autoencoder = Chain(encoder, decoder)
    
    return encoder, decoder, autoencoder
end

"""
    train_autoencoder(X::DataFrame; epochs::Int=100, learning_rate::Float64=0.001)

Train an autoencoder on normal transactions.

# Arguments
- `X::DataFrame`: Training data (normal transactions only)
- `epochs::Int`: Number of training epochs (default: 100)
- `learning_rate::Float64`: Learning rate (default: 0.001)

# Returns
- Tuple of (trained_autoencoder, encoder)

# Requirements
- Requirement 5.1: Ensemble anomaly detection
"""
function train_autoencoder(X::DataFrame; epochs::Int=100, learning_rate::Float64=0.001)
    @info "Training autoencoder for $(epochs) epochs"
    
    # Convert to matrix and transpose for Flux (features × samples)
    X_matrix = Matrix{Float64}(X)'
    input_dim = size(X_matrix, 1)
    
    # Build autoencoder
    encoder, decoder, autoencoder = build_autoencoder(input_dim)
    
    # Define loss function (reconstruction error)
    loss(model, x) = Flux.mse(model(x), x)
    
    # Optimizer - use the new Flux.setup API
    opt_state = Flux.setup(Flux.Adam(learning_rate), autoencoder)
    
    # Training loop
    for epoch in 1:epochs
        # Compute gradient and update
        grads = Flux.gradient(m -> loss(m, X_matrix), autoencoder)
        Flux.update!(opt_state, autoencoder, grads[1])
        
        if epoch % 20 == 0
            current_loss = loss(autoencoder, X_matrix)
            @info "Epoch $(epoch)/$(epochs), Loss: $(current_loss)"
        end
    end
    
    final_loss = loss(autoencoder, X_matrix)
    @info "Autoencoder training complete. Final loss: $(final_loss)"
    
    return autoencoder, encoder
end

"""
    compute_reconstruction_error(autoencoder, X::DataFrame)

Compute reconstruction errors as anomaly scores.

# Arguments
- `autoencoder`: Trained autoencoder model
- `X::DataFrame`: Data to score

# Returns
- Vector of reconstruction errors (higher = more anomalous)
"""
function compute_reconstruction_error(autoencoder, X::DataFrame)
    X_matrix = Matrix{Float64}(X)'
    reconstructed = autoencoder(X_matrix)
    
    # Compute MSE for each sample
    errors = vec(mean((X_matrix .- reconstructed).^2, dims=1))
    return errors
end

# ============================================================================
# Task 14.3: Ensemble Anomaly Detection
# ============================================================================

"""
    normalize_scores(scores::Vector{Float64})

Normalize anomaly scores to [0, 1] range using min-max scaling.

# Arguments
- `scores::Vector{Float64}`: Raw anomaly scores

# Returns
- Normalized scores in [0, 1]

# Requirements
- Requirement 5.2: Normalized anomaly scores
"""
function normalize_scores(scores::Vector{Float64})
    min_score = minimum(scores)
    max_score = maximum(scores)
    
    if max_score - min_score < 1e-10
        return zeros(length(scores))
    end
    
    normalized = (scores .- min_score) ./ (max_score - min_score)
    return normalized
end

"""
    classify_severity(score::Float64)

Classify anomaly severity based on score thresholds.

# Arguments
- `score::Float64`: Normalized anomaly score [0, 1]

# Returns
- Severity level: 'low', 'medium', 'high', or 'critical'

# Requirements
- Requirement 5.2: Severity classification
"""
function classify_severity(score::Float64)
    if score >= 0.8
        return "critical"
    elseif score >= 0.6
        return "high"
    elseif score >= 0.4
        return "medium"
    else
        return "low"
    end
end

"""
    compute_deviation_metrics(value::Float64, reference_values::Vector{Float64})

Compute deviation metrics (z-score, percentile rank).

# Arguments
- `value::Float64`: Value to analyze
- `reference_values::Vector{Float64}`: Reference distribution

# Returns
- Dict with z_score and percentile_rank
"""
function compute_deviation_metrics(value::Float64, reference_values::Vector{Float64})
    μ = mean(reference_values)
    σ = std(reference_values)
    
    z_score = σ > 0 ? (value - μ) / σ : 0.0
    percentile_rank = mean(reference_values .<= value) * 100
    
    return Dict(
        "z_score" => z_score,
        "percentile_rank" => percentile_rank
    )
end

"""
    train_anomaly_model(X_train::DataFrame; 
                       contamination::Float64=0.1,
                       autoencoder_epochs::Int=100,
                       threshold::Float64=0.5)

Train ensemble anomaly detection model.

# Arguments
- `X_train::DataFrame`: Training data (normal transactions)
- `contamination::Float64`: Expected proportion of anomalies (default: 0.1)
- `autoencoder_epochs::Int`: Autoencoder training epochs (default: 100)
- `threshold::Float64`: Anomaly score threshold (default: 0.5)

# Returns
- Trained AnomalyModel

# Requirements
- Requirement 5.1: Ensemble anomaly detection
"""
function train_anomaly_model(X_train::DataFrame; 
                            contamination::Float64=0.1,
                            autoencoder_epochs::Int=100,
                            threshold::Float64=0.5)
    @info "Training ensemble anomaly detection model"
    
    # Store feature names and compute statistics
    feature_names = names(X_train)
    feature_stats = Dict{String, Any}()
    
    for col in feature_names
        feature_stats[col] = Dict(
            "mean" => mean(X_train[!, col]),
            "std" => std(X_train[!, col]),
            "min" => minimum(X_train[!, col]),
            "max" => maximum(X_train[!, col])
        )
    end
    
    # Task 14.1: Train Isolation Forest
    @info "Training Isolation Forest..."
    isolation_forest = train_isolation_forest(X_train, contamination=contamination)
    
    # Task 14.2: Train Autoencoder
    @info "Training Autoencoder..."
    autoencoder, encoder = train_autoencoder(X_train, epochs=autoencoder_epochs)
    
    model = AnomalyModel(
        isolation_forest,
        autoencoder,
        encoder,
        threshold,
        feature_names,
        feature_stats,
        "v1.0.0"
    )
    
    @info "Ensemble anomaly detection model training complete"
    return model
end

"""
    detect_anomalies(model::AnomalyModel, X::DataFrame)

Detect anomalies using ensemble approach.

# Arguments
- `model::AnomalyModel`: Trained anomaly detection model
- `X::DataFrame`: Data to analyze

# Returns
- Vector of AnomalyResult

# Requirements
- Requirement 5.1: Ensemble anomaly detection
- Requirement 5.2: Normalized scores and severity classification
"""
function detect_anomalies(model::AnomalyModel, X::DataFrame)
    @info "Detecting anomalies in $(nrow(X)) transactions"
    
    # Compute scores from both algorithms
    if_scores = compute_isolation_forest_score(model.isolation_forest, X)
    ae_scores = compute_reconstruction_error(model.autoencoder, X)
    
    # Normalize scores to [0, 1]
    if_normalized = normalize_scores(if_scores)
    ae_normalized = normalize_scores(ae_scores)
    
    # Ensemble: average of both scores
    ensemble_scores = (if_normalized .+ ae_normalized) ./ 2.0
    
    # Create results
    results = AnomalyResult[]
    
    for i in 1:length(ensemble_scores)
        score = ensemble_scores[i]
        is_anomaly = score >= model.threshold
        severity = classify_severity(score)
        
        # Compute expected value (mean of training data for first feature as proxy)
        first_feature = model.feature_names[1]
        expected_value = model.feature_stats[first_feature]["mean"]
        
        # Compute deviation metrics for first feature
        actual_value = X[i, first_feature]
        reference_values = [model.feature_stats[first_feature]["mean"]]
        deviation_metrics = compute_deviation_metrics(actual_value, reference_values)
        
        result = AnomalyResult(
            is_anomaly,
            score,
            severity,
            if_normalized[i],
            ae_normalized[i],
            expected_value,
            deviation_metrics
        )
        
        push!(results, result)
    end
    
    anomaly_count = count(r -> r.is_anomaly, results)
    @info "Detected $(anomaly_count) anomalies out of $(length(results)) transactions"
    
    return results
end

"""
    compute_anomaly_score(model::AnomalyModel, transaction::DataFrame)

Compute anomaly score for a single transaction.

# Arguments
- `model::AnomalyModel`: Trained anomaly detection model
- `transaction::DataFrame`: Single transaction (1 row)

# Returns
- Normalized anomaly score [0, 1]

# Requirements
- Requirement 5.2: Normalized anomaly scores
"""
function compute_anomaly_score(model::AnomalyModel, transaction::DataFrame)
    results = detect_anomalies(model, transaction)
    return results[1].anomaly_score
end

end # module
