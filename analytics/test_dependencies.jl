#!/usr/bin/env julia

"""
Test script to verify all required dependencies are available.
This script attempts to load all dependencies specified in Project.toml.
"""

using Pkg

println("Testing Julia Analytics Engine Dependencies")
println("=" ^ 60)

# Activate the project environment
Pkg.activate(".")

println("\nInstantiating project dependencies...")
try
    Pkg.instantiate()
    println("✓ Dependencies instantiated successfully")
catch e
    println("✗ Failed to instantiate dependencies: $e")
    exit(1)
end

# Test loading each critical dependency
dependencies = [
    "MLJ" => "Machine Learning framework",
    "XGBoost" => "Gradient boosting",
    "Flux" => "Neural networks",
    "Clustering" => "Spatial clustering algorithms",
    "OutlierDetection" => "Anomaly detection",
    "ShapML" => "SHAP explainability",
    "LibPQ" => "PostGIS database connection",
    "gRPC" => "gRPC communication",
    "ProtoBuf" => "Protocol Buffers",
    "DataFrames" => "Data manipulation",
    "Statistics" => "Statistical functions",
    "StatsBase" => "Statistical utilities",
    "Distances" => "Distance metrics",
    "JSON3" => "JSON serialization",
    "Logging" => "Logging utilities"
]

println("\nTesting dependency loading:")
println("-" ^ 60)

all_loaded = true
for (pkg, description) in dependencies
    try
        eval(Meta.parse("using $pkg"))
        println("✓ $pkg - $description")
    catch e
        println("✗ $pkg - Failed to load: $e")
        all_loaded = false
    end
end

println("-" ^ 60)

if all_loaded
    println("\n✓ All dependencies loaded successfully!")
    println("\nTesting AnalyticsEngine module...")
    try
        include("src/AnalyticsEngine.jl")
        using .AnalyticsEngine
        println("✓ AnalyticsEngine module loaded successfully")
        exit(0)
    catch e
        println("✗ Failed to load AnalyticsEngine module: $e")
        exit(1)
    end
else
    println("\n✗ Some dependencies failed to load")
    exit(1)
end
