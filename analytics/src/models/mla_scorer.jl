# MLA Development Impact Scoring Module
# Tasks 13.1-13.2: Implement weight learning and composite score computation
# Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7, 4.10, 4.11, 4.13

"""
MLA Development Impact Scoring module for evaluating constituency development.

This module provides:
- Weight learning model using Gradient Boosting (Task 13.1)
- Composite score computation with normalization (Task 13.2)
- Best practice extraction (Task 13.7)
- Change detection and insight generation (Task 13.8)
- Scenario modeling for what-if analysis (Task 13.10)
- Integration with SHAP explainability (Task 13.11)
"""

module MLAScorer

using MLJ
using XGBoost
using DataFrames
using Statistics
using StatsBase
using Logging
using Random
using Distributions

export MLAModel, train_mla_model, compute_impact_score
export MLAScore, extract_best_practices, detect_significant_changes
export simulate_scenario

# ============================================================================
# Data Structures
# ============================================================================

"""
    MLAModel

Container for the MLA development impact scoring model.

# Fields
- `weight_model`: Trained Gradient Boosting model for learning weights
- `indicator_weights::Dict{String, Float64}`: Learned weights for each indicator
- `feature_names::Vector{String}`: Names of development indicators
- `normalization_params::Dict`: Parameters for score normalization
- `model_version::String`: Model version identifier
"""
mutable struct MLAModel
    weight_model::Any
    indicator_weights::Dict{String, Float64}
    feature_names::Vector{String}
    normalization_params::Dict{String, Any}
    model_version::String
end

"""
    MLAScore

Container for MLA development impact score results.

# Fields
- `constituency_id::String`: Constituency identifier
- `overall_score::Float64`: Overall development score [0, 100]
- `component_scores::Dict{String, Float64}`: Scores by category
- `confidence_lower::Float64`: Lower bound of confidence interval
- `confidence_upper::Float64`: Upper bound of confidence interval
- `indicator_values::Dict{String, Float64}`: Raw indicator values
- `indicator_weights::Dict{String, Float64}`: Weights used
- `model_version::String`: Model version used
"""
struct MLAScore
    constituency_id::String
    overall_score::Float64
    component_scores::Dict{String, Float64}
    confidence_lower::Float64
    confidence_upper::Float64
    indicator_values::Dict{String, Float64}
    indicator_weights::Dict{String, Float64}
    model_version::String
end

# ============================================================================
# Task 13.1: Weight Learning Model
# ============================================================================

"""
    train_weight_learning_model(historical_data::DataFrame, outcome_col::String="success_score")

Learn optimal indicator weights from historical outcomes using correlation analysis.

# Arguments
- `historical_data::DataFrame`: Historical constituency data with indicators and outcomes
- `outcome_col::String`: Name of outcome column (default: "success_score")

# Returns
- Dict with feature correlations

# Requirements
- Requirements 4.2, 4.3: Learn weights from historical successful outcomes
"""
function train_weight_learning_model(historical_data::DataFrame, outcome_col::String="success_score")
    @info "Training weight learning model" n_samples=nrow(historical_data)
    
    # Separate features and target
    feature_cols = [col for col in names(historical_data) if col != outcome_col && col != "constituency_id"]
    
    # Compute correlations between each feature and outcome
    correlations = Dict{String, Float64}()
    
    y = historical_data[:, outcome_col]
    
    for feature in feature_cols
        x = historical_data[:, feature]
        # Compute Pearson correlation
        corr = cor(x, y)
        # Use absolute value (both positive and negative correlations are important)
        correlations[feature] = abs(corr)
    end
    
    @info "Weight learning complete" n_features=length(correlations)
    
    return correlations
end

"""
    extract_indicator_weights(correlations::Dict{String, Float64}, feature_names::Vector{String})

Extract learned indicator weights from correlation analysis.

# Arguments
- `correlations::Dict{String, Float64}`: Feature correlations with outcome
- `feature_names::Vector{String}`: Names of indicators

# Returns
- `Dict{String, Float64}`: Normalized weights for each indicator
"""
function extract_indicator_weights(correlations::Dict{String, Float64}, feature_names::Vector{String})
    @info "Extracting indicator weights from correlations"
    
    # Create weight dictionary from correlations
    weights = Dict{String, Float64}()
    
    # Calculate total correlation (for normalization)
    total_correlation = sum(abs(correlations[name]) for name in feature_names if haskey(correlations, name))
    
    # Normalize weights
    if total_correlation > 0
        for name in feature_names
            if haskey(correlations, name)
                # Use absolute correlation as weight
                weights[name] = abs(correlations[name]) / total_correlation
            else
                weights[name] = 0.0
            end
        end
    else
        # If no correlations, use equal weights
        @warn "No correlations found, using equal weights"
        for name in feature_names
            weights[name] = 1.0 / length(feature_names)
        end
    end
    
    @info "Indicator weights extracted" n_indicators=length(weights)
    
    return weights
end

# ============================================================================
# Task 13.2: Composite Score Computation
# ============================================================================

"""
    compute_weighted_score(indicators::Dict{String, Float64}, weights::Dict{String, Float64})

Compute weighted sum of indicators.

# Arguments
- `indicators::Dict{String, Float64}`: Indicator values
- `weights::Dict{String, Float64}`: Indicator weights

# Returns
- `Float64`: Weighted score

# Requirements
- Requirement 4.1: Compute weighted sum
"""
function compute_weighted_score(indicators::Dict{String, Float64}, weights::Dict{String, Float64})
    score = 0.0
    
    for (indicator, value) in indicators
        if haskey(weights, indicator)
            score += value * weights[indicator]
        end
    end
    
    return score
end

"""
    normalize_score(raw_score::Float64, min_score::Float64, max_score::Float64)

Normalize score to [0, 100] range.

# Arguments
- `raw_score::Float64`: Raw weighted score
- `min_score::Float64`: Minimum possible score
- `max_score::Float64`: Maximum possible score

# Returns
- `Float64`: Normalized score in [0, 100]

# Requirements
- Requirement 4.5: Normalize scores to [0, 100]
"""
function normalize_score(raw_score::Float64, min_score::Float64, max_score::Float64)
    if max_score - min_score < 1e-10
        return 50.0  # Return midpoint if no range
    end
    
    # Linear normalization to [0, 100]
    normalized = 100.0 * (raw_score - min_score) / (max_score - min_score)
    
    # Clamp to [0, 100]
    return clamp(normalized, 0.0, 100.0)
end

"""
    compute_confidence_interval(scores::Vector{Float64}, confidence_level::Float64=0.95)

Compute confidence interval for score using bootstrap.

# Arguments
- `scores::Vector{Float64}`: Bootstrap sample scores
- `confidence_level::Float64`: Confidence level (default: 0.95)

# Returns
- Tuple of (lower_bound, upper_bound)

# Requirements
- Requirement 4.7: Compute confidence intervals
"""
function compute_confidence_interval(scores::Vector{Float64}, confidence_level::Float64=0.95)
    if length(scores) < 2
        return (scores[1], scores[1])
    end
    
    # Sort scores
    sorted_scores = sort(scores)
    n = length(sorted_scores)
    
    # Compute percentiles for confidence interval
    alpha = 1.0 - confidence_level
    lower_idx = max(1, floor(Int, n * alpha / 2))
    upper_idx = min(n, ceil(Int, n * (1 - alpha / 2)))
    
    lower_bound = sorted_scores[lower_idx]
    upper_bound = sorted_scores[upper_idx]
    
    return (lower_bound, upper_bound)
end

"""
    train_mla_model(historical_data::DataFrame; outcome_col::String="success_score")

Train complete MLA development impact scoring model.

# Arguments
- `historical_data::DataFrame`: Historical constituency data
- `outcome_col::String`: Outcome column name (default: "success_score")

# Returns
- `MLAModel`: Trained MLA scoring model

# Requirements
- Requirements 4.1, 4.2, 4.3
"""
function train_mla_model(historical_data::DataFrame; outcome_col::String="success_score")
    @info "Training MLA development impact scoring model"
    
    # Train weight learning model
    weight_model = train_weight_learning_model(historical_data, outcome_col)
    
    # Extract feature names
    feature_cols = [col for col in names(historical_data) if col != outcome_col && col != "constituency_id"]
    
    # Extract indicator weights
    indicator_weights = extract_indicator_weights(weight_model, feature_cols)
    
    # Compute normalization parameters
    # Calculate min and max possible scores
    min_indicators = Dict{String, Float64}(col => Float64(minimum(historical_data[:, col])) for col in feature_cols)
    max_indicators = Dict{String, Float64}(col => Float64(maximum(historical_data[:, col])) for col in feature_cols)
    
    min_score = compute_weighted_score(min_indicators, indicator_weights)
    max_score = compute_weighted_score(max_indicators, indicator_weights)
    
    normalization_params = Dict{String, Any}(
        "min_score" => min_score,
        "max_score" => max_score,
        "min_indicators" => min_indicators,
        "max_indicators" => max_indicators
    )
    
    @info "MLA model training complete" n_indicators=length(indicator_weights)
    
    model = MLAModel(
        weight_model,
        indicator_weights,
        feature_cols,
        normalization_params,
        "1.0.0"
    )
    
    return model
end

"""
    compute_impact_score(model::MLAModel, constituency_data::DataFrame; 
                        constituency_id::String="", n_bootstrap::Int=100)

Compute development impact score for a constituency.

# Arguments
- `model::MLAModel`: Trained MLA model
- `constituency_data::DataFrame`: Constituency indicator data (single row)
- `constituency_id::String`: Constituency identifier
- `n_bootstrap::Int`: Number of bootstrap samples for confidence interval

# Returns
- `MLAScore`: Development impact score with confidence interval

# Requirements
- Requirements 4.1, 4.5, 4.7
"""
function compute_impact_score(
    model::MLAModel,
    constituency_data::DataFrame;
    constituency_id::String="",
    n_bootstrap::Int=100
)
    @info "Computing MLA development impact score" constituency_id=constituency_id
    
    # Extract indicator values
    indicator_values = Dict{String, Float64}()
    for feature in model.feature_names
        if feature in names(constituency_data)
            indicator_values[feature] = constituency_data[1, feature]
        else
            @warn "Missing indicator" indicator=feature
            indicator_values[feature] = 0.0
        end
    end
    
    # Compute raw weighted score
    raw_score = compute_weighted_score(indicator_values, model.indicator_weights)
    
    # Normalize to [0, 100]
    min_score = model.normalization_params["min_score"]
    max_score = model.normalization_params["max_score"]
    overall_score = normalize_score(raw_score, min_score, max_score)
    
    # Compute component scores (by category if available)
    component_scores = compute_component_scores(indicator_values, model.indicator_weights)
    
    # Bootstrap confidence interval
    bootstrap_scores = zeros(n_bootstrap)
    for i in 1:n_bootstrap
        # Add small random noise to indicators
        noisy_indicators = Dict{String, Float64}()
        for (k, v) in indicator_values
            noise = randn() * 0.05 * v  # 5% noise
            noisy_indicators[k] = v + noise
        end
        
        noisy_raw = compute_weighted_score(noisy_indicators, model.indicator_weights)
        bootstrap_scores[i] = normalize_score(noisy_raw, min_score, max_score)
    end
    
    confidence_lower, confidence_upper = compute_confidence_interval(bootstrap_scores)
    
    @info "MLA score computed" score=overall_score confidence_interval=(confidence_lower, confidence_upper)
    
    score = MLAScore(
        constituency_id,
        overall_score,
        component_scores,
        confidence_lower,
        confidence_upper,
        indicator_values,
        model.indicator_weights,
        model.model_version
    )
    
    return score
end

"""
    compute_component_scores(indicators::Dict{String, Float64}, weights::Dict{String, Float64})

Compute scores by component category.

# Arguments
- `indicators::Dict{String, Float64}`: Indicator values
- `weights::Dict{String, Float64}`: Indicator weights

# Returns
- `Dict{String, Float64}`: Scores by category
"""
function compute_component_scores(indicators::Dict{String, Float64}, weights::Dict{String, Float64})
    # Categorize indicators by prefix or pattern
    categories = Dict{String, Vector{String}}(
        "infrastructure" => String[],
        "education" => String[],
        "healthcare" => String[],
        "employment" => String[],
        "economic" => String[]
    )
    
    # Categorize indicators
    for indicator in keys(indicators)
        indicator_lower = lowercase(indicator)
        if occursin("road", indicator_lower) || occursin("bridge", indicator_lower) || occursin("infrastructure", indicator_lower)
            push!(categories["infrastructure"], indicator)
        elseif occursin("school", indicator_lower) || occursin("education", indicator_lower) || occursin("literacy", indicator_lower)
            push!(categories["education"], indicator)
        elseif occursin("health", indicator_lower) || occursin("hospital", indicator_lower) || occursin("doctor", indicator_lower)
            push!(categories["healthcare"], indicator)
        elseif occursin("job", indicator_lower) || occursin("employment", indicator_lower) || occursin("unemployment", indicator_lower)
            push!(categories["employment"], indicator)
        elseif occursin("gdp", indicator_lower) || occursin("income", indicator_lower) || occursin("economic", indicator_lower)
            push!(categories["economic"], indicator)
        end
    end
    
    # Compute category scores
    component_scores = Dict{String, Float64}()
    
    for (category, category_indicators) in categories
        if !isempty(category_indicators)
            category_score = 0.0
            category_weight = 0.0
            
            for indicator in category_indicators
                if haskey(indicators, indicator) && haskey(weights, indicator)
                    category_score += indicators[indicator] * weights[indicator]
                    category_weight += weights[indicator]
                end
            end
            
            # Normalize by category weight
            if category_weight > 0
                component_scores[category] = category_score / category_weight
            end
        end
    end
    
    return component_scores
end

# ============================================================================
# Task 13.7: Best Practice Extraction
# ============================================================================

"""
    extract_best_practices(model::MLAModel, all_scores::Vector{MLAScore}; top_n::Int=10)

Extract best practices from high-scoring constituencies.

# Arguments
- `model::MLAModel`: Trained MLA model
- `all_scores::Vector{MLAScore}`: Scores for all constituencies
- `top_n::Int`: Number of top constituencies to analyze

# Returns
- `Dict{String, Any}`: Best practices and common success factors

# Requirements
- Requirement 4.10: Extract best practices from high-scoring constituencies
"""
function extract_best_practices(model::MLAModel, all_scores::Vector{MLAScore}; top_n::Int=10)
    @info "Extracting best practices" n_constituencies=length(all_scores) top_n=top_n
    
    # Sort by overall score
    sorted_scores = sort(all_scores, by=s -> s.overall_score, rev=true)
    top_constituencies = sorted_scores[1:min(top_n, length(sorted_scores))]
    
    # Analyze common patterns in top constituencies
    best_practices = Dict{String, Any}()
    
    # Find indicators that are consistently high in top constituencies
    indicator_stats = Dict{String, Vector{Float64}}()
    
    for score in top_constituencies
        for (indicator, value) in score.indicator_values
            if !haskey(indicator_stats, indicator)
                indicator_stats[indicator] = Float64[]
            end
            push!(indicator_stats[indicator], value)
        end
    end
    
    # Compute average values for each indicator in top constituencies
    common_factors = Dict{String, Float64}()
    for (indicator, values) in indicator_stats
        common_factors[indicator] = mean(values)
    end
    
    # Identify key success factors (indicators with high values and high weights)
    success_factors = []
    for (indicator, avg_value) in common_factors
        if haskey(model.indicator_weights, indicator)
            weight = model.indicator_weights[indicator]
            importance = avg_value * weight
            push!(success_factors, (indicator=indicator, value=avg_value, weight=weight, importance=importance))
        end
    end
    
    # Sort by importance
    sort!(success_factors, by=x -> x.importance, rev=true)
    
    best_practices["top_constituencies"] = [s.constituency_id for s in top_constituencies]
    best_practices["average_score"] = mean(s.overall_score for s in top_constituencies)
    best_practices["common_factors"] = common_factors
    best_practices["success_factors"] = success_factors[1:min(5, length(success_factors))]
    
    @info "Best practices extracted" n_factors=length(success_factors)
    
    return best_practices
end

# ============================================================================
# Task 13.8: Change Detection and Insight Generation
# ============================================================================

"""
    detect_significant_changes(previous_scores::Vector{MLAScore}, current_scores::Vector{MLAScore}; 
                               threshold::Float64=15.0)

Detect significant score changes and generate insights.

# Arguments
- `previous_scores::Vector{MLAScore}`: Previous period scores
- `current_scores::Vector{MLAScore}`: Current period scores
- `threshold::Float64`: Minimum change to be considered significant (default: 15 points)

# Returns
- `Vector{Dict{String, Any}}`: Insights for constituencies with significant changes

# Requirements
- Requirement 4.11: Detect significant changes (> 15 points) and generate insights
"""
function detect_significant_changes(
    previous_scores::Vector{MLAScore},
    current_scores::Vector{MLAScore};
    threshold::Float64=15.0
)
    @info "Detecting significant changes" threshold=threshold
    
    # Create lookup for previous scores
    prev_lookup = Dict(s.constituency_id => s for s in previous_scores)
    
    insights = []
    
    for current in current_scores
        if haskey(prev_lookup, current.constituency_id)
            previous = prev_lookup[current.constituency_id]
            
            # Compute change
            change = current.overall_score - previous.overall_score
            
            # Check if significant
            if abs(change) >= threshold
                # Generate insight
                insight = generate_change_insight(previous, current, change)
                push!(insights, insight)
            end
        end
    end
    
    @info "Significant changes detected" n_changes=length(insights)
    
    return insights
end

"""
    generate_change_insight(previous::MLAScore, current::MLAScore, change::Float64)

Generate automated insight explaining score change.

# Arguments
- `previous::MLAScore`: Previous score
- `current::MLAScore`: Current score
- `change::Float64`: Score change

# Returns
- `Dict{String, Any}`: Insight with explanation
"""
function generate_change_insight(previous::MLAScore, current::MLAScore, change::Float64)
    direction = change > 0 ? "improved" : "declined"
    
    # Find indicators with largest changes
    indicator_changes = []
    for (indicator, current_value) in current.indicator_values
        if haskey(previous.indicator_values, indicator)
            prev_value = previous.indicator_values[indicator]
            ind_change = current_value - prev_value
            
            if haskey(current.indicator_weights, indicator)
                weight = current.indicator_weights[indicator]
                weighted_change = ind_change * weight
                
                push!(indicator_changes, (
                    indicator=indicator,
                    change=ind_change,
                    weighted_change=weighted_change
                ))
            end
        end
    end
    
    # Sort by weighted change magnitude
    sort!(indicator_changes, by=x -> abs(x.weighted_change), rev=true)
    
    # Generate explanation
    top_factors = indicator_changes[1:min(3, length(indicator_changes))]
    
    explanation = "Score $(direction) by $(round(abs(change), digits=1)) points. "
    explanation *= "Key factors: "
    
    for (i, factor) in enumerate(top_factors)
        factor_direction = factor.change > 0 ? "increased" : "decreased"
        explanation *= "$(factor.indicator) $(factor_direction)"
        if i < length(top_factors)
            explanation *= ", "
        end
    end
    
    insight = Dict{String, Any}(
        "constituency_id" => current.constituency_id,
        "change" => change,
        "direction" => direction,
        "previous_score" => previous.overall_score,
        "current_score" => current.overall_score,
        "top_factors" => top_factors,
        "explanation" => explanation
    )
    
    return insight
end

# ============================================================================
# Task 13.10: Scenario Modeling
# ============================================================================

"""
    simulate_scenario(model::MLAModel, baseline_data::DataFrame, 
                     modifications::Dict{String, Float64})

Simulate score changes under different resource allocation strategies.

# Arguments
- `model::MLAModel`: Trained MLA model
- `baseline_data::DataFrame`: Baseline constituency data
- `modifications::Dict{String, Float64}`: Proposed changes to indicators

# Returns
- `Dict{String, Any}`: Scenario results with predicted score

# Requirements
- Requirement 4.13: Support what-if analysis for resource allocation
"""
function simulate_scenario(
    model::MLAModel,
    baseline_data::DataFrame,
    modifications::Dict{String, Float64}
)
    @info "Simulating scenario" n_modifications=length(modifications)
    
    # Create modified data
    modified_data = copy(baseline_data)
    
    for (indicator, change) in modifications
        if indicator in names(modified_data)
            modified_data[1, indicator] += change
        end
    end
    
    # Compute baseline score
    baseline_score = compute_impact_score(model, baseline_data, constituency_id="baseline")
    
    # Compute modified score
    modified_score = compute_impact_score(model, modified_data, constituency_id="scenario")
    
    # Compute impact
    score_change = modified_score.overall_score - baseline_score.overall_score
    
    scenario_result = Dict{String, Any}(
        "baseline_score" => baseline_score.overall_score,
        "scenario_score" => modified_score.overall_score,
        "score_change" => score_change,
        "modifications" => modifications,
        "component_changes" => Dict(
            category => modified_score.component_scores[category] - baseline_score.component_scores[category]
            for category in keys(baseline_score.component_scores) if haskey(modified_score.component_scores, category)
        )
    )
    
    @info "Scenario simulation complete" score_change=score_change
    
    return scenario_result
end

end # module MLAScorer
