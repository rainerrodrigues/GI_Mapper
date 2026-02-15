#!/usr/bin/env julia

"""
Test script for cluster detection integration

This script tests the complete clustering pipeline:
1. Load the spatial clusterer module
2. Create sample data points
3. Call the gRPC handler
4. Verify the response
"""

using Logging

# Set up logging
global_logger(ConsoleLogger(stdout, Logging.Info))

@info "Starting cluster integration test"

# Include the gRPC service module
include("src/grpc/service.jl")

# Create sample request data
sample_request = Dict(
    "data_points" => [
        Dict("id" => "p1", "longitude" => 77.2090, "latitude" => 28.6139, "value" => 100.0),
        Dict("id" => "p2", "longitude" => 77.2100, "latitude" => 28.6140, "value" => 150.0),
        Dict("id" => "p3", "longitude" => 77.2110, "latitude" => 28.6145, "value" => 120.0),
        Dict("id" => "p4", "longitude" => 77.3000, "latitude" => 28.7000, "value" => 200.0),
        Dict("id" => "p5", "longitude" => 77.3010, "latitude" => 28.7010, "value" => 180.0),
        Dict("id" => "p6", "longitude" => 77.3020, "latitude" => 28.7020, "value" => 190.0),
        Dict("id" => "p7", "longitude" => 77.1000, "latitude" => 28.5000, "value" => 80.0),
        Dict("id" => "p8", "longitude" => 77.1010, "latitude" => 28.5010, "value" => 90.0),
        Dict("id" => "p9", "longitude" => 77.1020, "latitude" => 28.5020, "value" => 85.0),
        Dict("id" => "p10", "longitude" => 77.1030, "latitude" => 28.5030, "value" => 95.0),
    ],
    "algorithm" => "auto"
)

@info "Testing cluster detection with $(length(sample_request["data_points"])) data points"

# Call the handler
try
    response = handle_detect_clusters(sample_request)
    
    @info "Cluster detection successful!"
    @info "Algorithm used: $(response["algorithm_used"])"
    @info "Number of clusters: $(length(response["clusters"]))"
    @info "Model version: $(response["model_version"])"
    
    # Display quality metrics
    metrics = response["quality_metrics"]
    @info "Quality Metrics:"
    @info "  Silhouette Score: $(metrics["silhouette_score"])"
    @info "  Davies-Bouldin Index: $(metrics["davies_bouldin_index"])"
    @info "  Calinski-Harabasz Score: $(metrics["calinski_harabasz_score"])"
    
    # Display cluster details
    for cluster in response["clusters"]
        @info "Cluster $(cluster["cluster_id"]):"
        @info "  Members: $(cluster["member_count"])"
        @info "  Centroid: $(cluster["centroid"])"
        @info "  Density: $(cluster["density"])"
        @info "  Economic Value: $(cluster["economic_value"])"
    end
    
    # Verify response structure
    @assert haskey(response, "clusters")
    @assert haskey(response, "algorithm_used")
    @assert haskey(response, "quality_metrics")
    @assert haskey(response, "model_version")
    @assert length(response["clusters"]) > 0
    
    @info "✓ All assertions passed!"
    @info "✓ Cluster integration test SUCCESSFUL"
    
catch e
    @error "Cluster detection failed" exception=e
    @error "✗ Cluster integration test FAILED"
    rethrow(e)
end
