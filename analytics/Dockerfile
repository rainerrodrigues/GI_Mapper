# Use Julia official image
FROM julia:1.10

WORKDIR /app

# Copy Project.toml and Manifest.toml
COPY analytics/Project.toml ./analytics/

# Install Julia dependencies
RUN julia --project=analytics -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy source code
COPY analytics/src ./analytics/src

# Expose gRPC port
EXPOSE 50051

# Run the analytics engine
CMD ["julia", "--project=analytics", "-e", "using AnalyticsEngine; AnalyticsEngine.start_service()"]
