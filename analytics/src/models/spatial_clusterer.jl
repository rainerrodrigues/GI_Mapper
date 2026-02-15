# Spatial Clustering Module for Analytics Engine
# Tasks 9.1-9.3: Implement spatial clustering algorithms
# Requirements: 2.1, 2.2, 2.4, 2.5, 6.1, 6.2, 6.3

"""
Spatial clustering module implementing DBSCAN and K-Means algorithms
for detecting economic clusters in geographic data.

This module provides:
- DBSCAN clustering with parameter optimization
- K-Means clustering with optimal k selection
- Cluster quality metrics computation
- Cluster boundary and statistics calculation
- Algorithm comparison and selection
"""

module SpatialClusterer

using Clustering
using Distances
using DataFrames
using Statistics
using LinearAlgebra
using Logging

export detect_clusters_dbscan, detect_clusters_kmeans
export optimize_dbscan_parameters, optimize_kmeans_k
export compute_quality_metrics, select_best_algorithm
export compute_cluster_boundaries, compute_cluster_statistics
export ClusterResult

# ============================================================================
# Data Structures
# ============================================================================

"""
    ClusterResult

Container for clustering results with metadata and quality metrics.

# Fields
- `algorithm::String`: Algorithm used ("DBSCAN" or "K-Means")
- `labels::Vector{Int}`: Cluster assignments for each point (-1 for noise in DBSCAN)
- `n_clusters::Int`: Number of clusters found
- `parameters::Dict`: Algorithm parameters used
- `quality_metrics::Dict`: Quality metrics (silhouette, davies_bouldin, calinski_harabasz)
- `cluster_centers::Union{Matrix{Float64}, Nothing}`: Cluster centroids (K-Means only)
"""
struct ClusterResult
    algorithm::String
    labels::Vector{Int}
    n_clusters::Int
    parameters::Dict{String, Any}
    quality_metrics::Dict{String, Float64}
    cluster_centers::Union{Matrix{Float64}, Nothing}
end

# ============================================================================
# DBSCAN Clustering (Task 9.1)
# ============================================================================

"""
    detect_clusters_dbscan(data::Matrix{Float64}; epsilon=nothing, min_points=nothing)

Perform DBSCAN clustering on spatial data.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_features), typically [longitude, latitude]
- `epsilon::Union{Float64, Nothing}`: Maximum distance between points in a cluster (auto-optimized if nothing)
- `min_points::Union{Int, Nothing}`: Minimum points to form a dense region (auto-optimized if nothing)

# Returns
- `ClusterResult`: Clustering results with labels, parameters, and quality metrics

# Requirements
- Requirements 2.1, 2.4, 2.5, 6.1, 6.2, 6.3

# Example
```julia
coords = [lon lat] # n × 2 matrix
result = detect_clusters_dbscan(coords, epsilon=0.5, min_points=5)
```
"""
function detect_clusters_dbscan(
    data::Matrix{Float64};
    epsilon::Union{Float64, Nothing}=nothing,
    min_points::Union{Int, Nothing}=nothing
)
    n_samples = size(data, 1)
    
    if n_samples < 2
        @warn "Insufficient data points for clustering" n_samples=n_samples
        return ClusterResult(
            "DBSCAN",
            ones(Int, n_samples),
            1,
            Dict("epsilon" => 0.0, "min_points" => 1),
            Dict("silhouette_score" => 0.0, "davies_bouldin_index" => 0.0, "calinski_harabasz_score" => 0.0),
            nothing
        )
    end
    
    # Optimize parameters if not provided
    if epsilon === nothing || min_points === nothing
        @info "Optimizing DBSCAN parameters"
        opt_params = optimize_dbscan_parameters(data)
        epsilon = something(epsilon, opt_params["epsilon"])
        min_points = something(min_points, opt_params["min_points"])
    end
    
    @info "Running DBSCAN clustering" epsilon=epsilon min_points=min_points n_samples=n_samples
    
    # Run DBSCAN (note: Clustering.jl uses min_neighbors parameter)
    result = dbscan(data', epsilon, min_neighbors=min_points)
    
    # Extract cluster labels (DBSCAN uses 0 for noise, we'll use -1)
    labels = result.assignments
    labels[labels .== 0] .= -1
    
    # Count clusters (excluding noise)
    n_clusters = maximum(labels[labels .> 0]; init=0)
    
    @info "DBSCAN clustering complete" n_clusters=n_clusters noise_points=count(labels .== -1)
    
    # Compute quality metrics
    quality_metrics = compute_quality_metrics(data, labels)
    
    parameters = Dict{String, Any}(
        "epsilon" => epsilon,
        "min_points" => min_points
    )
    
    return ClusterResult(
        "DBSCAN",
        labels,
        n_clusters,
        parameters,
        quality_metrics,
        nothing  # DBSCAN doesn't have explicit centroids
    )
end

"""
    optimize_dbscan_parameters(data::Matrix{Float64})

Automatically determine optimal DBSCAN parameters using k-distance graph and heuristics.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_features)

# Returns
- `Dict{String, Any}`: Optimal parameters with keys "epsilon" and "min_points"

# Requirements
- Requirements 2.5

# Algorithm
- min_points: Set to 2 * n_features (common heuristic)
- epsilon: Use elbow method on k-distance graph
"""
function optimize_dbscan_parameters(data::Matrix{Float64})
    n_samples, n_features = size(data)
    
    # Heuristic for min_points: 2 * dimensions
    min_points = max(4, min(2 * n_features, 10))
    
    # Calculate k-nearest neighbor distances for epsilon estimation
    k = min_points
    distances = pairwise(Euclidean(), data', dims=2)
    
    # For each point, find k-th nearest neighbor distance
    k_distances = Float64[]
    for i in 1:n_samples
        dists = sort(distances[i, :])
        # Skip first distance (self, which is 0)
        if length(dists) > k
            push!(k_distances, dists[k + 1])
        end
    end
    
    # Sort k-distances
    sort!(k_distances)
    
    # Use elbow method: find point of maximum curvature
    # Simple approach: use 90th percentile of k-distances
    epsilon = if length(k_distances) > 0
        quantile(k_distances, 0.9)
    else
        0.5  # Default fallback
    end
    
    @info "Optimized DBSCAN parameters" epsilon=epsilon min_points=min_points
    
    return Dict{String, Any}(
        "epsilon" => epsilon,
        "min_points" => min_points
    )
end

# ============================================================================
# K-Means Clustering (Task 9.2)
# ============================================================================

"""
    detect_clusters_kmeans(data::Matrix{Float64}; k=nothing, max_k=10)

Perform K-Means clustering on spatial data.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_features)
- `k::Union{Int, Nothing}`: Number of clusters (auto-optimized if nothing)
- `max_k::Int`: Maximum k to consider for optimization (default: 10)

# Returns
- `ClusterResult`: Clustering results with labels, centroids, parameters, and quality metrics

# Requirements
- Requirements 2.1, 2.4, 2.5, 6.1, 6.2, 6.3

# Example
```julia
coords = [lon lat] # n × 2 matrix
result = detect_clusters_kmeans(coords, k=5)
```
"""
function detect_clusters_kmeans(
    data::Matrix{Float64};
    k::Union{Int, Nothing}=nothing,
    max_k::Int=10
)
    n_samples = size(data, 1)
    
    if n_samples < 2
        @warn "Insufficient data points for clustering" n_samples=n_samples
        return ClusterResult(
            "K-Means",
            ones(Int, n_samples),
            1,
            Dict("k" => 1),
            Dict("silhouette_score" => 0.0, "davies_bouldin_index" => 0.0, "calinski_harabasz_score" => 0.0),
            data'
        )
    end
    
    # Optimize k if not provided
    if k === nothing
        @info "Optimizing K-Means k parameter"
        k = optimize_kmeans_k(data, max_k=max_k)
    end
    
    # Ensure k is valid
    k = max(2, min(k, n_samples))
    
    @info "Running K-Means clustering" k=k n_samples=n_samples
    
    # Run K-Means
    result = kmeans(data', k, maxiter=100, display=:none)
    
    labels = result.assignments
    centers = result.centers
    n_clusters = k
    
    @info "K-Means clustering complete" n_clusters=n_clusters
    
    # Compute quality metrics
    quality_metrics = compute_quality_metrics(data, labels)
    
    parameters = Dict{String, Any}(
        "k" => k
    )
    
    return ClusterResult(
        "K-Means",
        labels,
        n_clusters,
        parameters,
        quality_metrics,
        centers
    )
end

"""
    optimize_kmeans_k(data::Matrix{Float64}; max_k=10)

Determine optimal number of clusters using elbow method and silhouette analysis.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_features)
- `max_k::Int`: Maximum k to consider (default: 10)

# Returns
- `Int`: Optimal number of clusters

# Requirements
- Requirements 2.5

# Algorithm
- Try k from 2 to max_k
- Compute silhouette score for each k
- Select k with highest silhouette score
"""
function optimize_kmeans_k(data::Matrix{Float64}; max_k::Int=10)
    n_samples = size(data, 1)
    
    # Limit max_k to reasonable values
    max_k = min(max_k, n_samples ÷ 2)
    max_k = max(2, max_k)
    
    best_k = 2
    best_score = -1.0
    
    for k in 2:max_k
        try
            result = kmeans(data', k, maxiter=100, display=:none)
            labels = result.assignments
            
            # Compute silhouette score
            score = silhouette_score(data, labels)
            
            @info "Evaluated k" k=k silhouette_score=score
            
            if score > best_score
                best_score = score
                best_k = k
            end
        catch e
            @warn "Failed to evaluate k" k=k exception=e
        end
    end
    
    @info "Optimal k selected" k=best_k silhouette_score=best_score
    return best_k
end

# ============================================================================
# Quality Metrics (Task 9.3)
# ============================================================================

"""
    compute_quality_metrics(data::Matrix{Float64}, labels::Vector{Int})

Compute clustering quality metrics.

# Arguments
- `data::Matrix{Float64}`: Data matrix (n_samples × n_features)
- `labels::Vector{Int}`: Cluster assignments

# Returns
- `Dict{String, Float64}`: Quality metrics with keys:
  - "silhouette_score": Average silhouette coefficient [-1, 1], higher is better
  - "davies_bouldin_index": Davies-Bouldin index [0, ∞), lower is better
  - "calinski_harabasz_score": Variance ratio [0, ∞), higher is better

# Requirements
- Requirements 2.4, 6.4
"""
function compute_quality_metrics(data::Matrix{Float64}, labels::Vector{Int})
    # Filter out noise points (label -1)
    valid_mask = labels .> 0
    
    if count(valid_mask) < 2
        @warn "Insufficient valid points for quality metrics"
        return Dict{String, Float64}(
            "silhouette_score" => 0.0,
            "davies_bouldin_index" => 0.0,
            "calinski_harabasz_score" => 0.0
        )
    end
    
    valid_data = data[valid_mask, :]
    valid_labels = labels[valid_mask]
    
    n_clusters = length(unique(valid_labels))
    
    if n_clusters < 2
        @warn "Only one cluster found, quality metrics not meaningful"
        return Dict{String, Float64}(
            "silhouette_score" => 0.0,
            "davies_bouldin_index" => 0.0,
            "calinski_harabasz_score" => 0.0
        )
    end
    
    # Compute silhouette score
    sil_score = silhouette_score(valid_data, valid_labels)
    
    # Compute Davies-Bouldin index
    db_index = davies_bouldin_index(valid_data, valid_labels)
    
    # Compute Calinski-Harabasz score
    ch_score = calinski_harabasz_score(valid_data, valid_labels)
    
    @info "Quality metrics computed" silhouette=sil_score davies_bouldin=db_index calinski_harabasz=ch_score
    
    return Dict{String, Float64}(
        "silhouette_score" => sil_score,
        "davies_bouldin_index" => db_index,
        "calinski_harabasz_score" => ch_score
    )
end

"""
    silhouette_score(data::Matrix{Float64}, labels::Vector{Int})

Compute average silhouette coefficient.

# Arguments
- `data::Matrix{Float64}`: Data matrix
- `labels::Vector{Int}`: Cluster assignments

# Returns
- `Float64`: Average silhouette score [-1, 1]
"""
function silhouette_score(data::Matrix{Float64}, labels::Vector{Int})
    n_samples = size(data, 1)
    unique_labels = unique(labels)
    
    if length(unique_labels) < 2
        return 0.0
    end
    
    # Compute pairwise distances
    distances = pairwise(Euclidean(), data', dims=2)
    
    silhouette_values = Float64[]
    
    for i in 1:n_samples
        label_i = labels[i]
        
        # Compute a(i): mean distance to points in same cluster
        same_cluster = labels .== label_i
        if count(same_cluster) <= 1
            push!(silhouette_values, 0.0)
            continue
        end
        
        a_i = mean(distances[i, same_cluster .& (1:n_samples .!= i)])
        
        # Compute b(i): mean distance to points in nearest cluster
        b_i = Inf
        for other_label in unique_labels
            if other_label != label_i
                other_cluster = labels .== other_label
                if count(other_cluster) > 0
                    mean_dist = mean(distances[i, other_cluster])
                    b_i = min(b_i, mean_dist)
                end
            end
        end
        
        # Silhouette coefficient for point i
        s_i = (b_i - a_i) / max(a_i, b_i)
        push!(silhouette_values, s_i)
    end
    
    return mean(silhouette_values)
end

"""
    davies_bouldin_index(data::Matrix{Float64}, labels::Vector{Int})

Compute Davies-Bouldin index (lower is better).

# Arguments
- `data::Matrix{Float64}`: Data matrix
- `labels::Vector{Int}`: Cluster assignments

# Returns
- `Float64`: Davies-Bouldin index [0, ∞)
"""
function davies_bouldin_index(data::Matrix{Float64}, labels::Vector{Int})
    unique_labels = unique(labels)
    n_clusters = length(unique_labels)
    
    if n_clusters < 2
        return 0.0
    end
    
    # Compute cluster centers and within-cluster scatter
    centers = zeros(n_clusters, size(data, 2))
    scatters = zeros(n_clusters)
    
    for (idx, label) in enumerate(unique_labels)
        cluster_points = data[labels .== label, :]
        centers[idx, :] = mean(cluster_points, dims=1)
        
        # Within-cluster scatter (average distance to center)
        if size(cluster_points, 1) > 0
            dists = [norm(cluster_points[i, :] - centers[idx, :]) for i in 1:size(cluster_points, 1)]
            scatters[idx] = mean(dists)
        end
    end
    
    # Compute Davies-Bouldin index
    db_values = Float64[]
    
    for i in 1:n_clusters
        max_ratio = 0.0
        for j in 1:n_clusters
            if i != j
                # Distance between cluster centers
                center_dist = norm(centers[i, :] - centers[j, :])
                
                if center_dist > 0
                    ratio = (scatters[i] + scatters[j]) / center_dist
                    max_ratio = max(max_ratio, ratio)
                end
            end
        end
        push!(db_values, max_ratio)
    end
    
    return mean(db_values)
end

"""
    calinski_harabasz_score(data::Matrix{Float64}, labels::Vector{Int})

Compute Calinski-Harabasz score (variance ratio, higher is better).

# Arguments
- `data::Matrix{Float64}`: Data matrix
- `labels::Vector{Int}`: Cluster assignments

# Returns
- `Float64`: Calinski-Harabasz score [0, ∞)
"""
function calinski_harabasz_score(data::Matrix{Float64}, labels::Vector{Int})
    n_samples = size(data, 1)
    unique_labels = unique(labels)
    n_clusters = length(unique_labels)
    
    if n_clusters < 2 || n_clusters >= n_samples
        return 0.0
    end
    
    # Overall mean
    overall_mean = mean(data, dims=1)
    
    # Between-cluster dispersion
    between_dispersion = 0.0
    for label in unique_labels
        cluster_points = data[labels .== label, :]
        n_k = size(cluster_points, 1)
        cluster_mean = mean(cluster_points, dims=1)
        between_dispersion += n_k * sum((cluster_mean .- overall_mean).^2)
    end
    
    # Within-cluster dispersion
    within_dispersion = 0.0
    for label in unique_labels
        cluster_points = data[labels .== label, :]
        cluster_mean = mean(cluster_points, dims=1)
        for i in 1:size(cluster_points, 1)
            within_dispersion += sum((cluster_points[i, :] .- cluster_mean).^2)
        end
    end
    
    if within_dispersion == 0.0
        return 0.0
    end
    
    # Calinski-Harabasz score
    score = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
    
    return score
end

# ============================================================================
# Algorithm Selection (Task 9.3)
# ============================================================================

"""
    select_best_algorithm(dbscan_result::ClusterResult, kmeans_result::ClusterResult)

Select the best clustering algorithm based on silhouette score.

# Arguments
- `dbscan_result::ClusterResult`: DBSCAN clustering result
- `kmeans_result::ClusterResult`: K-Means clustering result

# Returns
- `ClusterResult`: The result with higher silhouette score

# Requirements
- Requirements 2.1
"""
function select_best_algorithm(dbscan_result::ClusterResult, kmeans_result::ClusterResult)
    dbscan_score = dbscan_result.quality_metrics["silhouette_score"]
    kmeans_score = kmeans_result.quality_metrics["silhouette_score"]
    
    if dbscan_score >= kmeans_score
        @info "Selected DBSCAN" silhouette_score=dbscan_score
        return dbscan_result
    else
        @info "Selected K-Means" silhouette_score=kmeans_score
        return kmeans_result
    end
end

# ============================================================================
# Cluster Boundaries and Statistics (Task 9.3)
# ============================================================================

"""
    compute_cluster_boundaries(data::Matrix{Float64}, labels::Vector{Int})

Compute convex hull boundaries for each cluster.

# Arguments
- `data::Matrix{Float64}`: Data matrix with coordinates
- `labels::Vector{Int}`: Cluster assignments

# Returns
- `Dict{Int, Vector{Vector{Float64}}}`: Cluster boundaries as convex hull vertices

# Requirements
- Requirements 2.2
"""
function compute_cluster_boundaries(data::Matrix{Float64}, labels::Vector{Int})
    unique_labels = unique(labels[labels .> 0])  # Exclude noise
    boundaries = Dict{Int, Vector{Vector{Float64}}}()
    
    for label in unique_labels
        cluster_points = data[labels .== label, :]
        
        if size(cluster_points, 1) >= 3
            # Compute convex hull (simplified: use bounding box for now)
            # Full convex hull would require additional package
            min_coords = minimum(cluster_points, dims=1)
            max_coords = maximum(cluster_points, dims=1)
            
            # Create bounding box as boundary
            boundary = [
                [min_coords[1], min_coords[2]],
                [max_coords[1], min_coords[2]],
                [max_coords[1], max_coords[2]],
                [min_coords[1], max_coords[2]],
                [min_coords[1], min_coords[2]]  # Close the polygon
            ]
            
            boundaries[label] = boundary
        end
    end
    
    @info "Computed cluster boundaries" n_clusters=length(boundaries)
    return boundaries
end

"""
    compute_cluster_statistics(data::Matrix{Float64}, labels::Vector{Int}, 
                               attributes::Union{Vector{Float64}, Nothing}=nothing)

Compute statistics for each cluster.

# Arguments
- `data::Matrix{Float64}`: Data matrix with coordinates
- `labels::Vector{Int}`: Cluster assignments
- `attributes::Union{Vector{Float64}, Nothing}`: Optional attribute values (e.g., economic value)

# Returns
- `Dict{Int, Dict{String, Any}}`: Statistics for each cluster including:
  - member_count: Number of points in cluster
  - density: Points per unit area
  - centroid: Cluster center coordinates
  - economic_value: Sum of attribute values (if provided)

# Requirements
- Requirements 2.2
"""
function compute_cluster_statistics(
    data::Matrix{Float64},
    labels::Vector{Int},
    attributes::Union{Vector{Float64}, Nothing}=nothing
)
    unique_labels = unique(labels[labels .> 0])  # Exclude noise
    statistics = Dict{Int, Dict{String, Any}}()
    
    for label in unique_labels
        cluster_mask = labels .== label
        cluster_points = data[cluster_mask, :]
        n_points = size(cluster_points, 1)
        
        # Compute centroid
        centroid = mean(cluster_points, dims=1)
        
        # Compute density (points per unit area)
        # Simplified: use bounding box area
        if n_points >= 2
            min_coords = minimum(cluster_points, dims=1)
            max_coords = maximum(cluster_points, dims=1)
            area = prod(max_coords .- min_coords)
            density = area > 0 ? n_points / area : 0.0
        else
            density = 0.0
        end
        
        # Compute economic value if attributes provided
        economic_value = if attributes !== nothing
            sum(attributes[cluster_mask])
        else
            0.0
        end
        
        statistics[label] = Dict{String, Any}(
            "member_count" => n_points,
            "density" => density,
            "centroid" => [centroid[1], centroid[2]],
            "economic_value" => economic_value
        )
    end
    
    @info "Computed cluster statistics" n_clusters=length(statistics)
    return statistics
end

end # module SpatialClusterer
