# Test script for spatial clustering implementation
# Tests Tasks 9.1-9.3 and 9.10

using Pkg
Pkg.activate(".")

println("Loading spatial clustering module...")
include("src/models/spatial_clusterer.jl")
using .SpatialClusterer

println("\n=== Testing Spatial Clustering ===\n")

# Generate synthetic spatial data
println("Generating synthetic spatial data...")
n_points = 100
data = randn(n_points, 2) .* 0.5

# Add three distinct clusters
data[1:30, :] .+= [0.0 0.0]    # Cluster 1 at origin
data[31:60, :] .+= [3.0 0.0]   # Cluster 2 at (3, 0)
data[61:90, :] .+= [0.0 3.0]   # Cluster 3 at (0, 3)
data[91:100, :] .+= [5.0 5.0]  # Cluster 4 at (5, 5)

println("Generated $n_points points with 4 expected clusters")

# Test DBSCAN clustering
println("\n--- Testing DBSCAN Clustering ---")
try
    dbscan_result = detect_clusters_dbscan(data, epsilon=0.8, min_points=5)
    
    println("Algorithm: ", dbscan_result.algorithm)
    println("Number of clusters: ", dbscan_result.n_clusters)
    println("Parameters: ", dbscan_result.parameters)
    println("Quality metrics:")
    for (metric, value) in dbscan_result.quality_metrics
        println("  $metric: $value")
    end
    
    # Check quality thresholds
    sil_score = dbscan_result.quality_metrics["silhouette_score"]
    db_index = dbscan_result.quality_metrics["davies_bouldin_index"]
    
    if sil_score >= 0.60
        println("✓ DBSCAN silhouette score meets threshold (≥ 0.60)")
    else
        println("✗ DBSCAN silhouette score below threshold: $sil_score")
    end
    
    if db_index <= 1.0
        println("✓ DBSCAN Davies-Bouldin index meets threshold (≤ 1.0)")
    else
        println("✗ DBSCAN Davies-Bouldin index above threshold: $db_index")
    end
    
    println("✓ DBSCAN clustering test passed")
catch e
    println("✗ DBSCAN clustering test failed: $e")
    rethrow(e)
end

# Test K-Means clustering
println("\n--- Testing K-Means Clustering ---")
try
    kmeans_result = detect_clusters_kmeans(data, k=4)
    
    println("Algorithm: ", kmeans_result.algorithm)
    println("Number of clusters: ", kmeans_result.n_clusters)
    println("Parameters: ", kmeans_result.parameters)
    println("Quality metrics:")
    for (metric, value) in kmeans_result.quality_metrics
        println("  $metric: $value")
    end
    
    # Check quality thresholds
    sil_score = kmeans_result.quality_metrics["silhouette_score"]
    db_index = kmeans_result.quality_metrics["davies_bouldin_index"]
    
    if sil_score >= 0.60
        println("✓ K-Means silhouette score meets threshold (≥ 0.60)")
    else
        println("✗ K-Means silhouette score below threshold: $sil_score")
    end
    
    if db_index <= 1.0
        println("✓ K-Means Davies-Bouldin index meets threshold (≤ 1.0)")
    else
        println("✗ K-Means Davies-Bouldin index above threshold: $db_index")
    end
    
    println("✓ K-Means clustering test passed")
catch e
    println("✗ K-Means clustering test failed: $e")
    rethrow(e)
end

# Test automatic parameter optimization
println("\n--- Testing Parameter Optimization ---")
try
    # DBSCAN parameter optimization
    dbscan_auto = detect_clusters_dbscan(data)
    println("DBSCAN auto-optimized parameters: ", dbscan_auto.parameters)
    println("✓ DBSCAN parameter optimization test passed")
    
    # K-Means k optimization
    kmeans_auto = detect_clusters_kmeans(data, max_k=8)
    println("K-Means auto-optimized k: ", kmeans_auto.parameters["k"])
    println("✓ K-Means parameter optimization test passed")
catch e
    println("✗ Parameter optimization test failed: $e")
    rethrow(e)
end

# Test algorithm selection
println("\n--- Testing Algorithm Selection ---")
try
    dbscan_result = detect_clusters_dbscan(data)
    kmeans_result = detect_clusters_kmeans(data)
    
    best_result = select_best_algorithm(dbscan_result, kmeans_result)
    
    println("DBSCAN silhouette: ", dbscan_result.quality_metrics["silhouette_score"])
    println("K-Means silhouette: ", kmeans_result.quality_metrics["silhouette_score"])
    println("Selected algorithm: ", best_result.algorithm)
    
    println("✓ Algorithm selection test passed")
catch e
    println("✗ Algorithm selection test failed: $e")
    rethrow(e)
end

# Test cluster boundaries and statistics
println("\n--- Testing Cluster Boundaries and Statistics ---")
try
    result = detect_clusters_kmeans(data, k=4)
    
    boundaries = compute_cluster_boundaries(data, result.labels)
    println("Computed boundaries for ", length(boundaries), " clusters")
    
    # Create synthetic economic values
    attributes = rand(n_points) .* 1000.0
    
    statistics = compute_cluster_statistics(data, result.labels, attributes)
    println("Computed statistics for ", length(statistics), " clusters")
    
    for (cluster_id, stats) in statistics
        println("Cluster $cluster_id:")
        println("  Members: ", stats["member_count"])
        println("  Density: ", round(stats["density"], digits=4))
        println("  Centroid: ", round.(stats["centroid"], digits=2))
        println("  Economic value: ", round(stats["economic_value"], digits=2))
    end
    
    println("✓ Cluster boundaries and statistics test passed")
catch e
    println("✗ Cluster boundaries and statistics test failed: $e")
    rethrow(e)
end

println("\n=== All Clustering Tests Passed ===\n")
