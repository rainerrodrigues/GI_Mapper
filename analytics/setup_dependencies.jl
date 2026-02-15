#!/usr/bin/env julia

"""
Setup script to install all required dependencies for the Analytics Engine.
Run this script from the analytics directory.
"""

using Pkg

println("Setting up Julia Analytics Engine Dependencies")
println("=" ^ 60)

# Activate the project environment
Pkg.activate(".")

# Core packages to install
packages = [
    # Machine Learning
    "MLJ",
    "XGBoost",
    "Flux",
    "Clustering",
    "OutlierDetection",
    "ShapML",
    
    # Data Processing
    "DataFrames",
    "CSV",
    "Statistics",
    "StatsBase",
    "Distances",
    
    # Database
    "LibPQ",
    
    # Communication (Note: gRPC.jl may need special handling)
    # "gRPC",
    "ProtoBuf",
    
    # Time Series
    "TimeSeries",
    "StateSpaceModels",
    
    # Utilities
    "Dates",
    "JSON3",
    "Logging"
]

println("\nInstalling packages...")
println("-" ^ 60)

for pkg in packages
    try
        println("Installing $pkg...")
        Pkg.add(pkg)
        println("✓ $pkg installed successfully")
    catch e
        println("✗ Failed to install $pkg: $e")
        println("  Continuing with other packages...")
    end
end

println("-" ^ 60)
println("\nResolving dependencies...")
try
    Pkg.resolve()
    println("✓ Dependencies resolved successfully")
catch e
    println("✗ Failed to resolve dependencies: $e")
end

println("\nInstantiating project...")
try
    Pkg.instantiate()
    println("✓ Project instantiated successfully")
catch e
    println("✗ Failed to instantiate project: $e")
end

println("\n" ^ "=" * 60)
println("Setup complete! Run test_dependencies.jl to verify.")
