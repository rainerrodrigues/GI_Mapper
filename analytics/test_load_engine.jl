#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

println("Loading AnalyticsEngine...")
using AnalyticsEngine

println("✓ AnalyticsEngine loaded successfully")
println("✓ SHAP module included")
println("Available modules:")
println("  - DataLoader")
println("  - FeatureEngineering")
println("  - SpatialClusterer")
println("  - SHAPEngine")
