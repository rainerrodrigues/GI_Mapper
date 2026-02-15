use axum::http::{header, Method};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;

/// Create the middleware stack for the application
pub fn create_middleware_stack() -> ServiceBuilder<
    tower::layer::util::Stack<
        TraceLayer<tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>>,
        CorsLayer,
    >,
> {
    ServiceBuilder::new()
        .layer(create_cors_layer())
        .layer(create_trace_layer())
}

/// Create CORS layer with appropriate configuration
fn create_cors_layer() -> CorsLayer {
    CorsLayer::new()
        // Allow requests from any origin in development
        // In production, this should be restricted to specific origins
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::ACCEPT,
        ])
        .allow_credentials(false)
}

/// Create tracing/logging layer
fn create_trace_layer() -> TraceLayer<tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>> {
    TraceLayer::new_for_http()
        .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
        .on_response(DefaultOnResponse::new().level(Level::INFO))
}
