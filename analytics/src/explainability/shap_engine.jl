# SHAP Explainability Engine for Analytics Engine
# Task 10.1: Create SHAP computation module
# Requirements: 7.1, 7.2, 7.4

"""
SHAP explainability module implementing SHAP value computation using ShapML
for explainable AI outputs across all model types.

This module provides:
- Local explanations for individual predictions
- Global explanations for overall model behavior
- Feature importance rankings
- Support for multiple model types (regression, classification, clustering)
- Formatted explanations for API responses
"""

module SHAPEngine

using ShapML
using DataFrames
using Statistics
using Logging
using Random
using StatsBase: sample

export compute_shap_local, compute_shap_global
export rank_features, format_explanation
export validate_shap_additivity
export generate_force_plot_data, generate_waterfall_plot_data
export generate_summary_plot_data, generate_dependence_plot_data
export generate_counterfactuals, compute_counterfactual_predictions
export format_counterfactual_explanation
export SHAPResult

# ============================================================================
# Data Structures
# ============================================================================

"""
    SHAPResult

Container for SHAP analysis results.

# Fields
- `feature_names::Vector{String}`: Names of features analyzed
- `shap_values::Vector{Float64}`: SHAP values for each feature
- `base_value::Float64`: Base/expected value of the model
- `prediction::Float64`: Model prediction for the instance
- `feature_values::Vector{Float64}`: Actual feature values for the instance
- `is_global::Bool`: Whether this is a global (true) or local (false) explanation
"""
struct SHAPResult
    feature_names::Vector{String}
    shap_values::Vector{Float64}
    base_value::Float64
    prediction::Float64
    feature_values::Vector{Float64}
    is_global::Bool
end

# ============================================================================
# Local Explanations (Task 10.1)
# ============================================================================

"""
    compute_shap_local(model, data::DataFrame, instance::DataFrame; 
                       sample_size::Int=60, target_col::String="target")

Compute SHAP values for a single prediction (local explanation).

Explains why the model made a specific prediction for an individual instance
by computing the contribution of each feature to the prediction.

# Arguments
- `model`: Trained model object (must support prediction interface)
- `data::DataFrame`: Reference dataset for computing SHAP values (background data)
- `instance::DataFrame`: Single instance to explain (1 row DataFrame)
- `sample_size::Int`: Number of Monte Carlo samples for SHAP estimation (default: 60)
- `target_col::String`: Name of target column (default: "target")

# Returns
- `SHAPResult`: SHAP values and metadata for the instance

# Requirements
- Requirements 7.1, 7.4

# Example
```julia
# Explain a single ROI prediction
instance = DataFrame(location=[28.6], sector=["agriculture"], amount=[100000])
shap_result = compute_shap_local(roi_model, training_data, instance)
```

# Performance
- Completes within 3 seconds for individual predictions (Requirement 7.6)
"""
function compute_shap_local(
    model,
    data::DataFrame,
    instance::DataFrame;
    sample_size::Int=60,
    target_col::String="target"
)
    @info "Computing local SHAP explanation" n_features=ncol(instance) sample_size=sample_size
    
    # Validate inputs
    if nrow(instance) != 1
        throw(ArgumentError("Instance must be a single row DataFrame"))
    end
    
    # Remove target column if present
    feature_cols = [col for col in names(data) if col != target_col]
    data_features = select(data, feature_cols)
    instance_features = select(instance, feature_cols)
    
    # Create prediction function wrapper
    function predict_function(model, data_df)
        # Handle different model types
        predictions = try
            # Try calling model directly
            model(data_df)
        catch e1
            try
                # Try predict method if available
                if isdefined(Main, :predict)
                    Main.predict(model, data_df)
                else
                    # Fallback: iterate over rows
                    [model(DataFrame(row)) for row in eachrow(data_df)]
                end
            catch e2
                @error "Prediction failed" exception=e2
                rethrow(e2)
            end
        end
        
        # Ensure we return a DataFrame with a single column
        return DataFrame(prediction = predictions)
    end
    
    # Compute SHAP values using ShapML
    # Use a sample of the reference data for efficiency
    reference_sample = if nrow(data_features) > sample_size
        data_features[sample(1:nrow(data_features), sample_size, replace=false), :]
    else
        data_features
    end
    
    try
        # Compute SHAP values
        shap_df = ShapML.shap(
            explain = instance_features,
            reference = reference_sample,
            model = model,
            predict_function = predict_function,
            sample_size = sample_size,
            seed = 1
        )
        
        # Extract SHAP values for each feature
        feature_names = String[]
        shap_values = Float64[]
        feature_values = Float64[]
        
        for col in feature_cols
            push!(feature_names, col)
            
            # Get SHAP value for this feature
            feature_shap = filter(row -> row.feature_name == col, shap_df)
            if nrow(feature_shap) > 0
                push!(shap_values, feature_shap[1, :shap_effect])
            else
                push!(shap_values, 0.0)
            end
            
            # Get actual feature value
            push!(feature_values, instance_features[1, col])
        end
        
        # Get prediction and base value
        prediction_df = predict_function(model, instance_features)
        prediction = prediction_df[1, :prediction]
        
        # Base value is the mean prediction on reference data
        base_predictions = predict_function(model, reference_sample)
        base_value = mean(base_predictions.prediction)
        
        @info "Local SHAP computation complete" n_features=length(feature_names) prediction=prediction
        
        return SHAPResult(
            feature_names,
            shap_values,
            base_value,
            prediction,
            feature_values,
            false  # is_global = false
        )
        
    catch e
        @error "SHAP computation failed" exception=e
        rethrow(e)
    end
end

# ============================================================================
# Global Explanations (Task 10.1)
# ============================================================================

"""
    compute_shap_global(model, data::DataFrame; 
                        sample_size::Int=100, target_col::String="target")

Compute SHAP values for overall model behavior (global explanation).

Explains which features are most important for the model's predictions overall
by computing average absolute SHAP values across multiple instances.

# Arguments
- `model`: Trained model object (must support prediction interface)
- `data::DataFrame`: Dataset to explain (will sample if too large)
- `sample_size::Int`: Number of instances to explain (default: 100)
- `target_col::String`: Name of target column (default: "target")

# Returns
- `SHAPResult`: Aggregated SHAP values showing global feature importance

# Requirements
- Requirements 7.1, 7.4

# Example
```julia
# Explain overall ROI model behavior
shap_result = compute_shap_global(roi_model, training_data)
```

# Performance
- Completes within 30 seconds for global explanations (Requirement 7.6)
"""
function compute_shap_global(
    model,
    data::DataFrame;
    sample_size::Int=100,
    target_col::String="target"
)
    @info "Computing global SHAP explanation" n_samples=nrow(data) sample_size=sample_size
    
    # Remove target column if present
    feature_cols = [col for col in names(data) if col != target_col]
    data_features = select(data, feature_cols)
    
    # Sample data if too large
    explain_sample = if nrow(data_features) > sample_size
        data_features[sample(1:nrow(data_features), sample_size, replace=false), :]
    else
        data_features
    end
    
    # Create prediction function wrapper
    function predict_function(model, data_df)
        predictions = try
            # Try calling model directly
            model(data_df)
        catch e1
            try
                # Try predict method if available
                if isdefined(Main, :predict)
                    Main.predict(model, data_df)
                else
                    # Fallback: iterate over rows
                    [model(DataFrame(row)) for row in eachrow(data_df)]
                end
            catch e2
                @error "Prediction failed" exception=e2
                rethrow(e2)
            end
        end
        return DataFrame(prediction = predictions)
    end
    
    try
        # Compute SHAP values for all instances
        shap_df = ShapML.shap(
            explain = explain_sample,
            reference = data_features,
            model = model,
            predict_function = predict_function,
            sample_size = 60,
            seed = 1
        )
        
        # Aggregate SHAP values by feature (mean absolute SHAP value)
        feature_names = String[]
        shap_values = Float64[]
        feature_values = Float64[]
        
        for col in feature_cols
            push!(feature_names, col)
            
            # Get all SHAP values for this feature
            feature_shap = filter(row -> row.feature_name == col, shap_df)
            if nrow(feature_shap) > 0
                # Use mean absolute SHAP value for global importance
                push!(shap_values, mean(abs.(feature_shap.shap_effect)))
                # Use mean feature value
                push!(feature_values, mean(feature_shap.feature_value))
            else
                push!(shap_values, 0.0)
                push!(feature_values, 0.0)
            end
        end
        
        # Get average prediction
        predictions = predict_function(model, explain_sample)
        avg_prediction = mean(predictions.prediction)
        
        # Base value is the mean prediction
        base_value = avg_prediction
        
        @info "Global SHAP computation complete" n_features=length(feature_names) avg_prediction=avg_prediction
        
        return SHAPResult(
            feature_names,
            shap_values,
            base_value,
            avg_prediction,
            feature_values,
            true  # is_global = true
        )
        
    catch e
        @error "Global SHAP computation failed" exception=e
        rethrow(e)
    end
end

# ============================================================================
# Feature Ranking (Task 10.1)
# ============================================================================

"""
    rank_features(shap_result::SHAPResult; top_k::Union{Int, Nothing}=nothing)

Rank features by their SHAP value importance.

# Arguments
- `shap_result::SHAPResult`: SHAP analysis result
- `top_k::Union{Int, Nothing}`: Return only top k features (default: all)

# Returns
- `DataFrame`: Ranked features with columns:
  - feature_name: Name of the feature
  - shap_value: SHAP value (absolute for global, signed for local)
  - feature_value: Actual feature value
  - rank: Importance rank (1 = most important)

# Requirements
- Requirements 7.2

# Example
```julia
ranked = rank_features(shap_result, top_k=10)
```
"""
function rank_features(shap_result::SHAPResult; top_k::Union{Int, Nothing}=nothing)
    # Use absolute SHAP values for ranking
    abs_shap = abs.(shap_result.shap_values)
    
    # Create ranking
    sorted_indices = sortperm(abs_shap, rev=true)
    
    # Limit to top_k if specified
    if top_k !== nothing
        sorted_indices = sorted_indices[1:min(top_k, length(sorted_indices))]
    end
    
    # Build result DataFrame
    ranked_df = DataFrame(
        rank = 1:length(sorted_indices),
        feature_name = shap_result.feature_names[sorted_indices],
        shap_value = shap_result.shap_values[sorted_indices],
        abs_shap_value = abs_shap[sorted_indices],
        feature_value = shap_result.feature_values[sorted_indices]
    )
    
    @info "Features ranked" n_features=nrow(ranked_df) top_feature=ranked_df[1, :feature_name]
    
    return ranked_df
end

# ============================================================================
# Explanation Formatting (Task 10.1)
# ============================================================================

"""
    format_explanation(shap_result::SHAPResult; top_k::Int=10)

Format SHAP explanation for API response.

Creates a structured explanation suitable for JSON serialization and
frontend visualization.

# Arguments
- `shap_result::SHAPResult`: SHAP analysis result
- `top_k::Int`: Number of top features to include (default: 10)

# Returns
- `Dict{String, Any}`: Formatted explanation with keys:
  - explanation_type: "local" or "global"
  - prediction: Model prediction value
  - base_value: Base/expected value
  - top_features: Array of top contributing features
  - all_features: Complete feature importance data
  - summary: Natural language summary

# Requirements
- Requirements 7.1, 7.2

# Example
```julia
explanation = format_explanation(shap_result, top_k=5)
# Returns JSON-serializable dictionary
```
"""
function format_explanation(shap_result::SHAPResult; top_k::Int=10)
    # Rank features
    ranked = rank_features(shap_result, top_k=nothing)
    
    # Get top features
    top_features = ranked[1:min(top_k, nrow(ranked)), :]
    
    # Format top features for API
    top_features_array = [
        Dict(
            "rank" => row.rank,
            "feature_name" => row.feature_name,
            "shap_value" => row.shap_value,
            "abs_shap_value" => row.abs_shap_value,
            "feature_value" => row.feature_value,
            "contribution" => row.shap_value > 0 ? "positive" : "negative"
        )
        for row in eachrow(top_features)
    ]
    
    # Format all features
    all_features_array = [
        Dict(
            "feature_name" => name,
            "shap_value" => shap_val,
            "feature_value" => feat_val
        )
        for (name, shap_val, feat_val) in zip(
            shap_result.feature_names,
            shap_result.shap_values,
            shap_result.feature_values
        )
    ]
    
    # Generate natural language summary
    summary = generate_summary(shap_result, top_features)
    
    # Build formatted explanation
    explanation = Dict{String, Any}(
        "explanation_type" => shap_result.is_global ? "global" : "local",
        "prediction" => shap_result.prediction,
        "base_value" => shap_result.base_value,
        "top_features" => top_features_array,
        "all_features" => all_features_array,
        "n_features" => length(shap_result.feature_names),
        "summary" => summary
    )
    
    @info "Explanation formatted" explanation_type=explanation["explanation_type"] n_top_features=length(top_features_array)
    
    return explanation
end

"""
    generate_summary(shap_result::SHAPResult, top_features::DataFrame)

Generate natural language summary of SHAP explanation.

Enhanced for Task 10.3 to provide context-aware summaries for different model types.

# Arguments
- `shap_result::SHAPResult`: SHAP analysis result
- `top_features::DataFrame`: Top ranked features

# Returns
- `String`: Natural language summary describing top 3 contributing factors

# Requirements
- Requirements 7.8 (natural language summaries)
"""
function generate_summary(shap_result::SHAPResult, top_features::DataFrame)
    if nrow(top_features) == 0
        return "No significant features identified."
    end
    
    # Get top 3 features
    n_top = min(3, nrow(top_features))
    top_3 = top_features[1:n_top, :]
    
    if shap_result.is_global
        # Global explanation summary
        summary_parts = String[]
        push!(summary_parts, "The model's predictions are primarily influenced by:")
        
        for (i, row) in enumerate(eachrow(top_3))
            feature = row.feature_name
            importance = round(row.abs_shap_value, digits=3)
            push!(summary_parts, "$(i). $(feature) (importance: $(importance))")
        end
        
        return join(summary_parts, " ")
    else
        # Local explanation summary
        summary_parts = String[]
        
        # Determine if prediction is above or below base
        diff = shap_result.prediction - shap_result.base_value
        direction = diff > 0 ? "higher" : "lower"
        
        push!(summary_parts, "This prediction is $(direction) than the base value.")
        push!(summary_parts, "The top contributing factors are:")
        
        for (i, row) in enumerate(eachrow(top_3))
            feature = row.feature_name
            shap_val = round(row.shap_value, digits=3)
            feat_val = round(row.feature_value, digits=3)
            effect = shap_val > 0 ? "increases" : "decreases"
            
            push!(summary_parts, "$(i). $(feature)=$(feat_val) $(effect) prediction by $(abs(shap_val))")
        end
        
        return join(summary_parts, " ")
    end
end

# ============================================================================
# Task 10.2: SHAP Visualization Data Generation
# ============================================================================

"""
    generate_force_plot_data(shap_result::SHAPResult)

Generate data for SHAP force plot visualization.

Force plots show how each feature pushes the prediction from the base value
to the final prediction value. Features that increase the prediction are shown
in red, features that decrease it are shown in blue.

# Arguments
- `shap_result::SHAPResult`: SHAP analysis result (must be local explanation)

# Returns
- `Dict{String, Any}`: Force plot data with keys:
  - base_value: Starting point (expected value)
  - prediction: Final prediction value
  - features: Array of features with their contributions
  - link: Type of link function ("identity" for regression)

# Requirements
- Requirements 7.3 (SHAP visualization types - force plots)

# Example
```julia
force_data = generate_force_plot_data(shap_result)
# Returns data suitable for frontend force plot visualization
```
"""
function generate_force_plot_data(shap_result::SHAPResult)
    if shap_result.is_global
        throw(ArgumentError("Force plots are only available for local explanations"))
    end
    
    # Sort features by absolute SHAP value for better visualization
    abs_shap = abs.(shap_result.shap_values)
    sorted_indices = sortperm(abs_shap, rev=true)
    
    # Build feature array
    features = [
        Dict(
            "name" => shap_result.feature_names[i],
            "value" => shap_result.feature_values[i],
            "shap_value" => shap_result.shap_values[i],
            "effect" => shap_result.shap_values[i] > 0 ? "positive" : "negative"
        )
        for i in sorted_indices
    ]
    
    force_plot_data = Dict{String, Any}(
        "base_value" => shap_result.base_value,
        "prediction" => shap_result.prediction,
        "features" => features,
        "link" => "identity",  # For regression; use "logit" for classification
        "visualization_type" => "force_plot"
    )
    
    @info "Force plot data generated" n_features=length(features)
    
    return force_plot_data
end

"""
    generate_waterfall_plot_data(shap_result::SHAPResult; top_k::Int=10)

Generate data for SHAP waterfall plot visualization.

Waterfall plots show the cumulative effect of features, starting from the base
value and showing how each feature contributes to reach the final prediction.

# Arguments
- `shap_result::SHAPResult`: SHAP analysis result (must be local explanation)
- `top_k::Int`: Number of top features to include (default: 10)

# Returns
- `Dict{String, Any}`: Waterfall plot data with keys:
  - base_value: Starting point
  - prediction: Final prediction
  - features: Array of features in order of contribution
  - cumulative_values: Cumulative sum at each step

# Requirements
- Requirements 7.3 (SHAP visualization types - waterfall plots)

# Example
```julia
waterfall_data = generate_waterfall_plot_data(shap_result, top_k=5)
```
"""
function generate_waterfall_plot_data(shap_result::SHAPResult; top_k::Int=10)
    if shap_result.is_global
        throw(ArgumentError("Waterfall plots are only available for local explanations"))
    end
    
    # Rank features by absolute SHAP value
    ranked = rank_features(shap_result, top_k=top_k)
    
    # Calculate cumulative values
    cumulative = shap_result.base_value
    cumulative_values = Float64[cumulative]
    
    features = []
    for row in eachrow(ranked)
        cumulative += row.shap_value
        push!(cumulative_values, cumulative)
        
        push!(features, Dict(
            "name" => row.feature_name,
            "value" => row.feature_value,
            "shap_value" => row.shap_value,
            "cumulative" => cumulative,
            "effect" => row.shap_value > 0 ? "positive" : "negative"
        ))
    end
    
    waterfall_data = Dict{String, Any}(
        "base_value" => shap_result.base_value,
        "prediction" => shap_result.prediction,
        "features" => features,
        "cumulative_values" => cumulative_values,
        "visualization_type" => "waterfall_plot"
    )
    
    @info "Waterfall plot data generated" n_features=length(features)
    
    return waterfall_data
end

"""
    generate_summary_plot_data(shap_results_array::Vector{SHAPResult})

Generate data for SHAP summary plot visualization.

Summary plots show the distribution of SHAP values for each feature across
multiple predictions, helping identify which features are consistently important.

# Arguments
- `shap_results_array::Vector{SHAPResult}`: Array of SHAP results from multiple predictions

# Returns
- `Dict{String, Any}`: Summary plot data with keys:
  - features: Array of feature names
  - shap_values: Matrix of SHAP values (features × instances)
  - feature_values: Matrix of feature values (features × instances)
  - mean_abs_shap: Mean absolute SHAP value per feature

# Requirements
- Requirements 7.3 (SHAP visualization types - summary plots)

# Example
```julia
# Compute SHAP for multiple instances
shap_results = [compute_shap_local(model, data, instance) for instance in instances]
summary_data = generate_summary_plot_data(shap_results)
```
"""
function generate_summary_plot_data(shap_results_array::Vector{SHAPResult})
    if isempty(shap_results_array)
        throw(ArgumentError("shap_results_array cannot be empty"))
    end
    
    # Check all results are local explanations
    if any(r.is_global for r in shap_results_array)
        throw(ArgumentError("Summary plots require local explanations only"))
    end
    
    # Get feature names from first result (assume all have same features)
    feature_names = shap_results_array[1].feature_names
    n_features = length(feature_names)
    n_instances = length(shap_results_array)
    
    # Build matrices of SHAP values and feature values
    shap_matrix = zeros(n_features, n_instances)
    feature_matrix = zeros(n_features, n_instances)
    
    for (j, result) in enumerate(shap_results_array)
        for (i, fname) in enumerate(feature_names)
            # Find index of this feature in the result
            idx = findfirst(==(fname), result.feature_names)
            if idx !== nothing
                shap_matrix[i, j] = result.shap_values[idx]
                feature_matrix[i, j] = result.feature_values[idx]
            end
        end
    end
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = [mean(abs.(shap_matrix[i, :])) for i in 1:n_features]
    
    # Sort features by importance
    sorted_indices = sortperm(mean_abs_shap, rev=true)
    
    # Build feature data
    features_data = [
        Dict(
            "name" => feature_names[i],
            "mean_abs_shap" => mean_abs_shap[i],
            "shap_values" => shap_matrix[i, :],
            "feature_values" => feature_matrix[i, :],
            "min_shap" => minimum(shap_matrix[i, :]),
            "max_shap" => maximum(shap_matrix[i, :]),
            "std_shap" => std(shap_matrix[i, :])
        )
        for i in sorted_indices
    ]
    
    summary_data = Dict{String, Any}(
        "features" => features_data,
        "n_instances" => n_instances,
        "n_features" => n_features,
        "visualization_type" => "summary_plot"
    )
    
    @info "Summary plot data generated" n_features=n_features n_instances=n_instances
    
    return summary_data
end

"""
    generate_dependence_plot_data(model, data::DataFrame, feature_name::String;
                                  target_col::String="target", sample_size::Int=100)

Generate data for SHAP dependence plot visualization.

Dependence plots show how a feature's value affects its SHAP value, revealing
the relationship between feature values and their impact on predictions.

# Arguments
- `model`: Trained model
- `data::DataFrame`: Reference dataset
- `feature_name::String`: Name of feature to analyze
- `target_col::String`: Target column name (default: "target")
- `sample_size::Int`: Number of instances to sample (default: 100)

# Returns
- `Dict{String, Any}`: Dependence plot data with keys:
  - feature_name: Name of the feature
  - feature_values: Array of feature values
  - shap_values: Corresponding SHAP values
  - interaction_feature: Feature with strongest interaction (optional)

# Requirements
- Requirements 7.3 (SHAP visualization types - dependence plots)

# Example
```julia
dependence_data = generate_dependence_plot_data(model, data, "investment_amount")
```
"""
function generate_dependence_plot_data(
    model,
    data::DataFrame,
    feature_name::String;
    target_col::String="target",
    sample_size::Int=100
)
    # Validate feature exists
    feature_cols = [col for col in names(data) if col != target_col]
    if !(feature_name in feature_cols)
        throw(ArgumentError("Feature '$feature_name' not found in data"))
    end
    
    # Sample data if too large
    sample_data = if nrow(data) > sample_size
        data[sample(1:nrow(data), sample_size, replace=false), :]
    else
        data
    end
    
    # Compute SHAP values for each instance
    feature_values = Float64[]
    shap_values = Float64[]
    
    for i in 1:nrow(sample_data)
        instance = sample_data[i:i, :]
        
        try
            shap_result = compute_shap_local(model, data, instance, 
                                            sample_size=60, target_col=target_col)
            
            # Find the SHAP value for our feature
            idx = findfirst(==(feature_name), shap_result.feature_names)
            if idx !== nothing
                push!(feature_values, shap_result.feature_values[idx])
                push!(shap_values, shap_result.shap_values[idx])
            end
        catch e
            @warn "Failed to compute SHAP for instance $i" exception=e
        end
    end
    
    # Find interaction feature (feature with highest correlation with SHAP values)
    # This is a simplified approach; full interaction detection is more complex
    interaction_feature = nothing
    max_correlation = 0.0
    
    for other_feature in feature_cols
        if other_feature != feature_name && other_feature in names(sample_data)
            other_values = [sample_data[i, other_feature] for i in 1:length(feature_values)]
            if length(other_values) == length(shap_values)
                corr = abs(cor(other_values, shap_values))
                if corr > max_correlation
                    max_correlation = corr
                    interaction_feature = other_feature
                end
            end
        end
    end
    
    dependence_data = Dict{String, Any}(
        "feature_name" => feature_name,
        "feature_values" => feature_values,
        "shap_values" => shap_values,
        "n_points" => length(feature_values),
        "visualization_type" => "dependence_plot"
    )
    
    if interaction_feature !== nothing && max_correlation > 0.3
        dependence_data["interaction_feature"] = interaction_feature
        dependence_data["interaction_correlation"] = max_correlation
    end
    
    @info "Dependence plot data generated" feature=feature_name n_points=length(feature_values)
    
    return dependence_data
end

# ============================================================================
# Task 10.4: Counterfactual Explanations
# ============================================================================

"""
    generate_counterfactuals(model, data::DataFrame, instance::DataFrame, 
                            target_change::Float64; n_counterfactuals::Int=5,
                            max_features_changed::Int=3, target_col::String="target")

Generate counterfactual scenarios showing how to achieve a target prediction change.

Counterfactuals answer "what-if" questions by showing how changing input features
would alter the prediction.

# Arguments
- `model`: Trained model
- `data::DataFrame`: Reference dataset for feasible feature ranges
- `instance::DataFrame`: Original instance to generate counterfactuals for
- `target_change::Float64`: Desired change in prediction (positive or negative)
- `n_counterfactuals::Int`: Number of counterfactuals to generate (default: 5)
- `max_features_changed::Int`: Maximum features to change per counterfactual (default: 3)
- `target_col::String`: Target column name (default: "target")

# Returns
- `Vector{DataFrame}`: Array of counterfactual instances

# Requirements
- Requirements 7.9 (counterfactual explanations)

# Example
```julia
# Generate counterfactuals to increase ROI by 10%
counterfactuals = generate_counterfactuals(model, data, instance, 0.10)
```
"""
function generate_counterfactuals(
    model,
    data::DataFrame,
    instance::DataFrame,
    target_change::Float64;
    n_counterfactuals::Int=5,
    max_features_changed::Int=3,
    target_col::String="target"
)
    if nrow(instance) != 1
        throw(ArgumentError("Instance must be a single row DataFrame"))
    end
    
    # Get feature columns
    feature_cols = [col for col in names(data) if col != target_col]
    
    # Get original prediction
    original_pred = try
        pred_df = DataFrame(prediction = predict(model, select(instance, feature_cols)))
        pred_df[1, :prediction]
    catch
        model(instance[1, :])
    end
    
    target_pred = original_pred + target_change
    
    @info "Generating counterfactuals" original_pred=original_pred target_pred=target_pred
    
    # Compute SHAP values to identify most influential features
    shap_result = compute_shap_local(model, data, instance, target_col=target_col)
    ranked = rank_features(shap_result, top_k=nothing)
    
    # Get feature ranges from data
    feature_ranges = Dict{String, Tuple{Float64, Float64}}()
    for col in feature_cols
        if eltype(data[!, col]) <: Number
            feature_ranges[col] = (minimum(data[!, col]), maximum(data[!, col]))
        end
    end
    
    counterfactuals = DataFrame[]
    attempts = 0
    max_attempts = n_counterfactuals * 10
    
    while length(counterfactuals) < n_counterfactuals && attempts < max_attempts
        attempts += 1
        
        # Create a copy of the instance
        cf = copy(instance)
        
        # Select features to modify (prioritize high SHAP value features)
        n_changes = rand(1:max_features_changed)
        features_to_change = ranked[1:min(n_changes, nrow(ranked)), :feature_name]
        
        # Modify selected features
        for feat in features_to_change
            if haskey(feature_ranges, feat)
                min_val, max_val = feature_ranges[feat]
                original_val = instance[1, feat]
                shap_val = ranked[ranked.feature_name .== feat, :shap_value][1]
                
                # Move feature in direction that helps achieve target
                if (target_change > 0 && shap_val > 0) || (target_change < 0 && shap_val < 0)
                    # Increase feature value
                    new_val = min(max_val, original_val + rand() * (max_val - original_val))
                else
                    # Decrease feature value
                    new_val = max(min_val, original_val - rand() * (original_val - min_val))
                end
                
                cf[1, feat] = new_val
            end
        end
        
        # Check if this counterfactual achieves desired change
        cf_pred = try
            pred_df = DataFrame(prediction = predict(model, select(cf, feature_cols)))
            pred_df[1, :prediction]
        catch
            model(cf[1, :])
        end
        
        # Accept if prediction is closer to target
        improvement = abs(cf_pred - target_pred) < abs(original_pred - target_pred)
        if improvement
            push!(counterfactuals, cf)
        end
    end
    
    @info "Counterfactuals generated" n_generated=length(counterfactuals) attempts=attempts
    
    return counterfactuals
end

"""
    compute_counterfactual_predictions(model, counterfactuals::Vector{DataFrame};
                                      target_col::String="target")

Compute predictions for counterfactual scenarios.

# Arguments
- `model`: Trained model
- `counterfactuals::Vector{DataFrame}`: Array of counterfactual instances
- `target_col::String`: Target column name (default: "target")

# Returns
- `Vector{Float64}`: Predictions for each counterfactual

# Requirements
- Requirements 7.9 (counterfactual explanations)

# Example
```julia
predictions = compute_counterfactual_predictions(model, counterfactuals)
```
"""
function compute_counterfactual_predictions(
    model,
    counterfactuals::Vector{DataFrame};
    target_col::String="target"
)
    predictions = Float64[]
    
    for cf in counterfactuals
        # Get feature columns
        feature_cols = [col for col in names(cf) if col != target_col]
        cf_features = select(cf, feature_cols)
        
        # Compute prediction
        pred = try
            pred_df = DataFrame(prediction = predict(model, cf_features))
            pred_df[1, :prediction]
        catch
            model(cf[1, :])
        end
        
        push!(predictions, pred)
    end
    
    @info "Counterfactual predictions computed" n_predictions=length(predictions)
    
    return predictions
end

"""
    format_counterfactual_explanation(original::DataFrame, counterfactuals::Vector{DataFrame},
                                     original_pred::Float64, cf_predictions::Vector{Float64};
                                     target_col::String="target")

Format counterfactual explanation for API response.

# Arguments
- `original::DataFrame`: Original instance
- `counterfactuals::Vector{DataFrame}`: Generated counterfactual instances
- `original_pred::Float64`: Original prediction
- `cf_predictions::Vector{Float64}`: Predictions for counterfactuals
- `target_col::String`: Target column name (default: "target")

# Returns
- `Dict{String, Any}`: Formatted counterfactual explanation

# Requirements
- Requirements 7.9 (counterfactual explanations)

# Example
```julia
explanation = format_counterfactual_explanation(original, counterfactuals, 
                                               original_pred, cf_predictions)
```
"""
function format_counterfactual_explanation(
    original::DataFrame,
    counterfactuals::Vector{DataFrame},
    original_pred::Float64,
    cf_predictions::Vector{Float64};
    target_col::String="target"
)
    # Get feature columns
    feature_cols = [col for col in names(original) if col != target_col]
    
    # Build counterfactual array
    cf_array = []
    
    for (i, (cf, pred)) in enumerate(zip(counterfactuals, cf_predictions))
        # Identify changed features
        changes = []
        for feat in feature_cols
            original_val = original[1, feat]
            cf_val = cf[1, feat]
            
            if original_val != cf_val
                push!(changes, Dict(
                    "feature" => feat,
                    "original_value" => original_val,
                    "counterfactual_value" => cf_val,
                    "change" => cf_val - original_val,
                    "change_pct" => original_val != 0 ? (cf_val - original_val) / original_val * 100 : 0.0
                ))
            end
        end
        
        push!(cf_array, Dict(
            "id" => i,
            "prediction" => pred,
            "prediction_change" => pred - original_pred,
            "prediction_change_pct" => original_pred != 0 ? (pred - original_pred) / original_pred * 100 : 0.0,
            "changes" => changes,
            "n_features_changed" => length(changes)
        ))
    end
    
    # Sort by prediction change magnitude
    sort!(cf_array, by = x -> abs(x["prediction_change"]), rev=true)
    
    explanation = Dict{String, Any}(
        "original_prediction" => original_pred,
        "counterfactuals" => cf_array,
        "n_counterfactuals" => length(cf_array),
        "explanation_type" => "counterfactual"
    )
    
    @info "Counterfactual explanation formatted" n_counterfactuals=length(cf_array)
    
    return explanation
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    validate_shap_additivity(shap_result::SHAPResult; tolerance::Float64=1e-6)

Validate that SHAP values satisfy the additivity property.

SHAP values should sum to the difference between prediction and base value.

# Arguments
- `shap_result::SHAPResult`: SHAP analysis result
- `tolerance::Float64`: Numerical tolerance for validation (default: 1e-6)

# Returns
- `Bool`: True if additivity property holds within tolerance

# Requirements
- Requirements 7.10 (SHAP additivity property)
"""
function validate_shap_additivity(shap_result::SHAPResult; tolerance::Float64=1e-6)
    # For local explanations, SHAP values should sum to prediction - base_value
    if !shap_result.is_global
        shap_sum = sum(shap_result.shap_values)
        expected_sum = shap_result.prediction - shap_result.base_value
        
        difference = abs(shap_sum - expected_sum)
        is_valid = difference <= tolerance
        
        if !is_valid
            @warn "SHAP additivity property violated" shap_sum=shap_sum expected_sum=expected_sum difference=difference
        else
            @info "SHAP additivity property validated" difference=difference
        end
        
        return is_valid
    else
        # Global explanations use mean absolute values, additivity doesn't apply
        return true
    end
end

"""
    compute_shap_for_tabular(model, data::DataFrame, instance::Union{DataFrame, Nothing}=nothing;
                             is_global::Bool=false, sample_size::Int=60, target_col::String="target")

Convenience function for computing SHAP values on tabular data.

# Arguments
- `model`: Trained model
- `data::DataFrame`: Reference/training data
- `instance::Union{DataFrame, Nothing}`: Instance to explain (for local) or nothing (for global)
- `is_global::Bool`: Whether to compute global explanation (default: false)
- `sample_size::Int`: Sample size for SHAP estimation
- `target_col::String`: Target column name

# Returns
- `SHAPResult`: SHAP analysis result
"""
function compute_shap_for_tabular(
    model,
    data::DataFrame,
    instance::Union{DataFrame, Nothing}=nothing;
    is_global::Bool=false,
    sample_size::Int=60,
    target_col::String="target"
)
    if is_global || instance === nothing
        return compute_shap_global(model, data, sample_size=sample_size, target_col=target_col)
    else
        return compute_shap_local(model, data, instance, sample_size=sample_size, target_col=target_col)
    end
end

"""
    compute_shap_for_spatial(model, data::DataFrame, instance::Union{DataFrame, Nothing}=nothing;
                            is_global::Bool=false, sample_size::Int=60, target_col::String="target")

Convenience function for computing SHAP values on spatial data.

Handles spatial features (latitude, longitude) along with other attributes.

# Arguments
- `model`: Trained model
- `data::DataFrame`: Reference/training data with spatial coordinates
- `instance::Union{DataFrame, Nothing}`: Instance to explain (for local) or nothing (for global)
- `is_global::Bool`: Whether to compute global explanation (default: false)
- `sample_size::Int`: Sample size for SHAP estimation
- `target_col::String`: Target column name

# Returns
- `SHAPResult`: SHAP analysis result
"""
function compute_shap_for_spatial(
    model,
    data::DataFrame,
    instance::Union{DataFrame, Nothing}=nothing;
    is_global::Bool=false,
    sample_size::Int=60,
    target_col::String="target"
)
    # Spatial data is handled the same way as tabular data
    # The spatial coordinates (lat, lon) are just additional features
    return compute_shap_for_tabular(model, data, instance, 
                                    is_global=is_global, 
                                    sample_size=sample_size, 
                                    target_col=target_col)
end

end # module SHAPEngine
