#!/usr/bin/env julia

"""
Simple test for spatial clustering module

This tests the clustering algorithms directly without the gRPC service layer.
"""

using Logging

# Set up logging
global_logger(ConsoleLogger(stdout, Logging.Info))

@info "Starting simple clustering test"

# Load the spatial clusterer module
include("src/models/spatial_clusterer.jl")
using .SpatialClusterer

# Create sample data (10 points in 2D space)
# Three clusters: around (77.2, 28.6), (77.3, 28.7), and (77.1, 28.5)
data = [
    77.2090 28.6139;
    77.2100 28.6140;
    77.2110 28.6145;
    77.3000 28.7000;
    77.3010 28.7010;
    77.3020 28.7020;
    77.1000 28.5000;
    77.1010 28.5010;
    77.1020 28.5020;
    77.1030 28.5030;
]

attributes = [100.0, 150.0, 120.0, 200.0, 180.0, 190.0, 80.0, 90.0, 85.0, 95.0]

@info "Testing DBSCAN clustering"
dbscan_result = detect_clusters_dbscan(data)
@info "DBSCAN found $(dbscan_result.n_clusters) clusters"
@info "DBSCAN silhouette score: $(dbscan_result.quality_metrics["silhouette_score"])"

@info "Testing K-Means clustering"
kmeans_result = detect_clusters_kmeans(data, k=3)
@info "K-Means found $(kmeans_result.n_clusters) clusters"
@info "K-Means silhouette score: $(kmeans_result.quality_metrics["silhouette_score"])"

@info "Testing algorithm selection"
best_result = select_best_algorithm(dbscan_result, kmeans_result)
@info "Best algorithm: $(best_result.algorithm)"

@info "Computing cluster boundaries"
boundaries = compute_cluster_boundaries(data, best_result.labels)
@info "Computed boundaries for $(length(boundaries)) clusters"

@info "Computing cluster statistics"
statistics = compute_cluster_statistics(data, best_result.labels, attributes)
@info "Computed statistics for $(length(statistics)) clusters"

# Display results
for cluster_id in sort(collect(keys(statistics)))
    @info "Cluster $cluster_id:"
    @info "  Members: $(statistics[cluster_id]["member_count"])"
    @info "  Centroid: $(statistics[cluster_id]["centroid"])"
    @info "  Density: $(statistics[cluster_id]["density"])"
    @info "  Economic Value: $(statistics[cluster_id]["economic_value"])"
end

@info "✓ Simple clustering test SUCCESSFUL"
