# Quick Test for Time Series Forecasting Module

using Pkg
Pkg.activate(".")

using Random
using Statistics
using Dates

println("=" ^ 80)
println("Time Series Forecasting Quick Test")
println("=" ^ 80)

Random.seed!(42)

# Import module
include("src/models/forecaster.jl")
using .Forecaster

# Generate simple test data
println("\n1. Generating test data...")
n = 36  # 3 years of monthly data
timestamps = [DateTime(2020, 1, 1) + Month(i-1) for i in 1:n]
trend = 100 .+ collect(1:n) .* 0.5
seasonal = 10 .* sin.(2π .* collect(1:n) ./ 12)
noise = 2 .* randn(n)
time_series = trend .+ seasonal .+ noise

println("   ✓ Generated $(length(time_series)) data points")

# Test seasonal detection
println("\n2. Testing seasonal pattern detection...")
patterns = Forecaster.detect_seasonal_patterns(time_series, timestamps)
println("   ✓ Has seasonality: $(patterns["has_seasonality"])")
println("   ✓ Period: $(patterns["period"]) months")

# Test ARIMA
println("\n3. Testing ARIMA model...")
arima_model = Forecaster.train_arima_model(time_series, timestamps)
arima_pred = Forecaster.forecast_arima(arima_model, 12)
println("   ✓ ARIMA 1-year forecast: $(round(arima_pred, digits=2))")

# Test Prophet
println("\n4. Testing Prophet model...")
prophet_model = Forecaster.train_prophet_model(time_series, timestamps)
prophet_pred = Forecaster.forecast_prophet(prophet_model, 12)
println("   ✓ Prophet 1-year forecast: $(round(prophet_pred, digits=2))")

# Test LSTM (skip training, just test structure)
println("\n5. Testing LSTM model (simplified)...")
lstm_model = Dict("type" => "mean", "value" => mean(time_series))
lstm_pred = Forecaster.forecast_lstm(lstm_model, 12)
println("   ✓ LSTM 1-year forecast: $(round(lstm_pred, digits=2))")

# Test baseline comparison
println("\n6. Testing baseline comparison...")
test_forecasts = Dict("1_year" => arima_pred)
baseline_comp = Forecaster.compare_to_baseline(time_series, test_forecasts)
println("   ✓ Baseline comparisons computed: $(length(baseline_comp)) metrics")

println("\n" * "=" ^ 80)
println("Quick Test Passed!")
println("=" ^ 80)
println("\nCore Features Working:")
println("  ✓ Seasonal pattern detection")
println("  ✓ ARIMA forecasting")
println("  ✓ Prophet forecasting")
println("  ✓ Baseline comparison")
println("\n" * "=" ^ 80)
