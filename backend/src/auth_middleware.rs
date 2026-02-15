use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::auth::AuthUser;
use crate::error::AppError;
use crate::models::UserRole;

/// Middleware to require authentication
pub async fn require_auth(
    auth_user: Result<AuthUser, AppError>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    // Ensure user is authenticated
    auth_user?;
    
    // Continue to the next handler
    Ok(next.run(request).await)
}

/// Middleware to require administrator role
pub async fn require_admin(
    auth_user: Result<AuthUser, AppError>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let user = auth_user?;
    
    if user.role != UserRole::Administrator {
        return Err(AppError::Forbidden(
            "Administrator role required".to_string()
        ));
    }
    
    Ok(next.run(request).await)
}

/// Middleware to require analyst role or higher
pub async fn require_analyst(
    auth_user: Result<AuthUser, AppError>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let user = auth_user?;
    
    if !user.role.has_permission(&UserRole::Analyst) {
        return Err(AppError::Forbidden(
            "Analyst role or higher required".to_string()
        ));
    }
    
    Ok(next.run(request).await)
}

/// Middleware to require viewer role or higher (any authenticated user)
pub async fn require_viewer(
    auth_user: Result<AuthUser, AppError>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let user = auth_user?;
    
    if !user.role.has_permission(&UserRole::Viewer) {
        return Err(AppError::Forbidden(
            "Viewer role or higher required".to_string()
        ));
    }
    
    Ok(next.run(request).await)
}
