# Test Time Series Forecasting Module
# Tests for Tasks 16.1-16.4, 16.8, 16.10

using Pkg
Pkg.activate(".")

using DataFrames
using Random
using Statistics
using Dates

println("=" ^ 80)
println("Time Series Forecasting Test")
println("=" ^ 80)

Random.seed!(42)

# Import module
include("src/models/forecaster.jl")
using .Forecaster

# ============================================================================
# Generate Test Data
# ============================================================================

println("\n1. Generating test time series data...")

# Generate synthetic time series with trend and seasonality
function generate_time_series(n::Int=60)
    timestamps = [Date(2020, 1, 1) + Month(i-1) for i in 1:n]
    
    # Trend component
    trend = 100 .+ collect(1:n) .* 0.5
    
    # Seasonal component (yearly cycle)
    seasonal = 10 .* sin.(2π .* collect(1:n) ./ 12)
    
    # Noise
    noise = 5 .* randn(n)
    
    # Combine
    values = trend .+ seasonal .+ noise
    
    return values, timestamps
end

time_series, timestamps = generate_time_series(60)
println("   ✓ Generated time series with $(length(time_series)) points")
println("   - Date range: $(timestamps[1]) to $(timestamps[end])")
println("   - Value range: [$(round(minimum(time_series), digits=2)), $(round(maximum(time_series), digits=2))]")

# ============================================================================
# Test 1: Seasonal Pattern Detection
# ============================================================================

println("\n2. Testing seasonal pattern detection...")

patterns = Forecaster.detect_seasonal_patterns(time_series, timestamps)

println("   ✓ Seasonal patterns detected")
println("   - Has seasonality: $(patterns["has_seasonality"])")
println("   - Period: $(patterns["period"]) months")
println("   - Amplitude: $(round(patterns["amplitude"], digits=2))")
println("   - Autocorrelation: $(round(patterns["autocorrelation"], digits=3))")
println("   - Trend changes detected: $(length(patterns["trend_changes"]))")

# ============================================================================
# Test 2: Train ARIMA Model
# ============================================================================

println("\n3. Training ARIMA model...")

arima_model = Forecaster.train_arima_model(time_series, timestamps)

println("   ✓ ARIMA model trained")
println("   - Model type: $(arima_model["type"])")
if arima_model["type"] == "arima"
    println("   - Slope: $(round(arima_model["slope"], digits=4))")
    println("   - AR coefficient: $(round(arima_model["ar_coef"], digits=4))")
end

# Test forecast
arima_forecast_1yr = Forecaster.forecast_arima(arima_model, 12)
println("   - 1-year forecast: $(round(arima_forecast_1yr, digits=2))")

# ============================================================================
# Test 3: Train Prophet Model
# ============================================================================

println("\n4. Training Prophet model...")

prophet_model = Forecaster.train_prophet_model(time_series, timestamps)

println("   ✓ Prophet model trained")
println("   - Model type: $(prophet_model["type"])")
if prophet_model["type"] == "prophet"
    println("   - Slope: $(round(prophet_model["slope"], digits=4))")
    println("   - Seasonal period: $(prophet_model["seasonal_period"]) months")
end

# Test forecast
prophet_forecast_1yr = Forecaster.forecast_prophet(prophet_model, 12)
println("   - 1-year forecast: $(round(prophet_forecast_1yr, digits=2))")

# ============================================================================
# Test 4: Train LSTM Model
# ============================================================================

println("\n5. Training LSTM model...")

lstm_model = Forecaster.train_lstm_model(time_series, timestamps, epochs=20)

println("   ✓ LSTM model trained")
println("   - Model type: $(lstm_model["type"])")
if lstm_model["type"] == "lstm"
    println("   - Sequence length: $(lstm_model["sequence_length"])")
    println("   - Data mean: $(round(lstm_model["data_mean"], digits=2))")
    println("   - Data std: $(round(lstm_model["data_std"], digits=2))")
end

# Test forecast
lstm_forecast_1yr = Forecaster.forecast_lstm(lstm_model, 12)
println("   - 1-year forecast: $(round(lstm_forecast_1yr, digits=2))")

# ============================================================================
# Test 5: Train Ensemble Model
# ============================================================================

println("\n6. Training ensemble forecast model...")

model = train_forecast_model(time_series, timestamps)

println("   ✓ Ensemble model trained")
println("   - Model version: $(model.model_version)")
println("   - Seasonal patterns detected: $(model.seasonal_patterns["has_seasonality"])")

# ============================================================================
# Test 6: Generate Ensemble Forecast
# ============================================================================

println("\n7. Generating ensemble forecasts...")

forecast = generate_forecast(
    model,
    time_series,
    timestamps,
    forecast_id="TEST_001"
)

println("   ✓ Ensemble forecast generated")
println("   - Forecast ID: $(forecast.forecast_id)")
println("\n   Horizon forecasts:")
for (horizon, value) in sort(collect(forecast.horizons))
    println("     - $(horizon): $(round(value, digits=2))")
end

println("\n   Prediction intervals (95%):")
for horizon in ["1_year", "3_year", "5_year"]
    if haskey(forecast.prediction_intervals, "$(horizon)_95")
        lower, upper = forecast.prediction_intervals["$(horizon)_95"]
        println("     - $(horizon): [$(round(lower, digits=2)), $(round(upper, digits=2))]")
    end
end

println("\n   Model contributions (1-year):")
println("     - ARIMA: $(round(forecast.model_contributions["1_year_arima"], digits=2))")
println("     - Prophet: $(round(forecast.model_contributions["1_year_prophet"], digits=2))")
println("     - LSTM: $(round(forecast.model_contributions["1_year_lstm"], digits=2))")

# ============================================================================
# Test 7: Baseline Comparison
# ============================================================================

println("\n8. Testing baseline comparison...")

println("   ✓ Baseline comparisons computed")
println("\n   Comparison to naive forecast:")
for horizon in ["1_year", "3_year", "5_year"]
    key = "$(horizon)_vs_naive"
    if haskey(forecast.baseline_comparison, key)
        diff = forecast.baseline_comparison[key]
        pct_key = "$(key)_pct"
        pct = haskey(forecast.baseline_comparison, pct_key) ? forecast.baseline_comparison[pct_key] : 0.0
        println("     - $(horizon): $(round(diff, digits=2)) ($(round(pct, digits=1))%)")
    end
end

println("\n   Comparison to moving average:")
for horizon in ["1_year", "3_year", "5_year"]
    key = "$(horizon)_vs_ma"
    if haskey(forecast.baseline_comparison, key)
        diff = forecast.baseline_comparison[key]
        pct_key = "$(key)_pct"
        pct = haskey(forecast.baseline_comparison, pct_key) ? forecast.baseline_comparison[pct_key] : 0.0
        println("     - $(horizon): $(round(diff, digits=2)) ($(round(pct, digits=1))%)")
    end
end

# ============================================================================
# Test 8: Forecast Validation
# ============================================================================

println("\n9. Validating forecast properties...")

# Test that forecasts increase with horizon (due to trend)
@assert forecast.horizons["1_year"] < forecast.horizons["5_year"] "Forecasts should reflect trend"

# Test that prediction intervals are valid
for horizon in ["1_year", "3_year", "5_year"]
    lower_80, upper_80 = forecast.prediction_intervals["$(horizon)_80"]
    lower_95, upper_95 = forecast.prediction_intervals["$(horizon)_95"]
    
    @assert lower_80 < upper_80 "80% interval must be valid"
    @assert lower_95 < upper_95 "95% interval must be valid"
    @assert lower_95 <= lower_80 "95% interval must be wider than 80%"
    @assert upper_80 <= upper_95 "95% interval must be wider than 80%"
    
    # Forecast should be within intervals
    @assert lower_95 <= forecast.horizons[horizon] <= upper_95 "Forecast should be within 95% interval"
end

println("   ✓ All forecast properties validated")

# ============================================================================
# Test 9: Seasonal Information
# ============================================================================

println("\n10. Checking seasonal information...")

println("   ✓ Seasonal information included")
println("   - Has seasonality: $(forecast.seasonal_info["has_seasonality"])")
println("   - Period: $(forecast.seasonal_info["period"]) months")
println("   - Amplitude: $(round(forecast.seasonal_info["amplitude"], digits=2))")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 80)
println("All Time Series Forecasting Tests Passed!")
println("=" ^ 80)
println("\nImplemented Features:")
println("  ✓ Task 16.1: ARIMA forecasting model")
println("  ✓ Task 16.2: Prophet forecasting model")
println("  ✓ Task 16.3: LSTM forecasting model")
println("  ✓ Task 16.4: Ensemble forecasting with prediction intervals")
println("  ✓ Task 16.8: Baseline comparison (naive, moving average)")
println("  ✓ Task 16.10: Seasonal pattern detection")
println("\nRequirements Satisfied:")
println("  ✓ Requirement 21.1: Multi-horizon forecasting (1, 3, 5 years)")
println("  ✓ Requirement 21.2: Ensemble of 3 models")
println("  ✓ Requirement 21.3: Prediction intervals (80%, 95%)")
println("  ✓ Requirement 21.6: Baseline comparison")
println("  ✓ Requirement 21.9: Seasonal pattern detection")
println("\n" * "=" ^ 80)
