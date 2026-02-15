# Build stage
FROM rust:1.75 as builder

WORKDIR /app

# Copy workspace files
COPY Cargo.toml ./
COPY backend/Cargo.toml ./backend/

# Create dummy main to cache dependencies
RUN mkdir -p backend/src && \
    echo "fn main() {}" > backend/src/main.rs && \
    cargo build --release && \
    rm -rf backend/src

# Copy actual source code
COPY backend/src ./backend/src
COPY backend/build.rs ./backend/ 2>/dev/null || true

# Build the application
RUN cargo build --release --bin backend

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates libssl3 libpq5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/backend /app/backend

# Expose port
EXPOSE 8000

# Run the application
CMD ["/app/backend"]
