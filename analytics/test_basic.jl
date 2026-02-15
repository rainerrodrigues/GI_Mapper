#!/usr/bin/env julia

"""
Basic test to verify the Analytics Engine module loads correctly.
"""

println("Testing AnalyticsEngine module...")
println("=" ^ 60)

try
    # Load the module
    include("src/AnalyticsEngine.jl")
    using .AnalyticsEngine
    
    println("✓ AnalyticsEngine module loaded successfully")
    
    # Test the start_service function
    println("\nTesting start_service()...")
    AnalyticsEngine.start_service()
    
    println("\n" * "=" ^ 60)
    println("✓ All tests passed!")
    exit(0)
catch e
    println("✗ Test failed: $e")
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end
