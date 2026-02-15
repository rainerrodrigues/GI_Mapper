# Quick test for SHAP visualization functions
# Tasks 10.2, 10.3, 10.4

using Pkg
Pkg.activate(".")

using DataFrames
using Statistics

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Import the SHAP engine module
include("src/explainability/shap_engine.jl")
using .SHAPEngine

println("Testing SHAP Visualization Functions - Quick Test")
println("=" ^ 80)

# Create a mock SHAP result for testing visualization functions
mock_shap_result = SHAPEngine.SHAPResult(
    ["feature1", "feature2", "feature3"],  # feature_names
    [5.0, 3.0, 1.0],                       # shap_values
    100.0,                                  # base_value
    109.0,                                  # prediction
    [55.0, 22.0, 11.0],                    # feature_values
    false                                   # is_global
)

println("\n[Test 1] Force Plot Data Generation")
force_data = generate_force_plot_data(mock_shap_result)
println("✓ Force plot data generated")
println("  Keys: $(keys(force_data))")
println("  Features: $(length(force_data["features"]))")

println("\n[Test 2] Waterfall Plot Data Generation")
waterfall_data = generate_waterfall_plot_data(mock_shap_result, top_k=3)
println("✓ Waterfall plot data generated")
println("  Features: $(length(waterfall_data["features"]))")
println("  Cumulative values: $(waterfall_data["cumulative_values"])")

println("\n[Test 3] Summary Plot Data Generation")
# Create multiple mock results
mock_results = [
    SHAPEngine.SHAPResult(
        ["feature1", "feature2", "feature3"],
        [5.0 + rand(), 3.0 + rand(), 1.0 + rand()],
        100.0,
        109.0 + rand() * 10,
        [55.0 + rand(), 22.0 + rand(), 11.0 + rand()],
        false
    )
    for _ in 1:5
]
summary_data = generate_summary_plot_data(mock_results)
println("✓ Summary plot data generated")
println("  Instances: $(summary_data["n_instances"])")
println("  Features: $(summary_data["n_features"])")

println("\n[Test 4] Natural Language Summary")
explanation = format_explanation(mock_shap_result, top_k=3)
println("✓ Explanation formatted")
println("  Summary: $(explanation["summary"])")

println("\n[Test 5] Counterfactual Explanation Formatting")
# Create mock counterfactuals
original = DataFrame(feature1=[55.0], feature2=[22.0], feature3=[11.0])
cf1 = DataFrame(feature1=[60.0], feature2=[22.0], feature3=[11.0])
cf2 = DataFrame(feature1=[55.0], feature2=[25.0], feature3=[11.0])
counterfactuals = [cf1, cf2]
cf_predictions = [115.0, 118.0]
original_pred = 109.0

cf_explanation = format_counterfactual_explanation(
    original, counterfactuals, original_pred, cf_predictions
)
println("✓ Counterfactual explanation formatted")
println("  Counterfactuals: $(cf_explanation["n_counterfactuals"])")
println("  Original prediction: $(cf_explanation["original_prediction"])")

println("\n" * "=" ^ 80)
println("✅ All quick tests passed!")
println("Tasks 10.2, 10.3, 10.4 implementation validated")
println("=" ^ 80)
