use axum::{routing::post, Router, Json, extract::State};
use serde_json::json;

use crate::auth::{hash_password, verify_password, generate_token, AuthUser};
use crate::db::{audit, user, Database};
use crate::error::AppError;
use crate::models::{LoginRequest, LoginResponse, UserInfo};

/// Authentication routes
pub fn routes() -> Router<Database> {
    Router::new()
        .route("/login", post(login))
        .route("/logout", post(logout))
        .route("/verify", post(verify))
}

/// Login handler
/// 
/// Authenticates a user with username and password, returns JWT token
/// 
/// # Requirements
/// * 14.1: Require authentication via username and password
/// * 14.2: Issue JWT token with expiration time on successful authentication
/// * 14.6: Log all authentication attempts with timestamps and IP addresses
async fn login(
    State(db): State<Database>,
    Json(request): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AppError> {
    // Find user by username
    let user = user::find_by_username(db.pool(), &request.username)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?
        .ok_or_else(|| {
            // Log failed authentication attempt
            let _ = tokio::spawn({
                let pool = db.pool().clone();
                let username = request.username.clone();
                async move {
                    let _ = audit::log_auth_attempt(&pool, None, "login_failed", None, false).await;
                    tracing::warn!("Failed login attempt for username: {}", username);
                }
            });
            AppError::Unauthorized("Invalid username or password".to_string())
        })?;

    // Verify password
    let password_valid = verify_password(&request.password, &user.password_hash)
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    if !password_valid {
        // Log failed authentication attempt
        let _ = tokio::spawn({
            let pool = db.pool().clone();
            let user_id = user.id;
            async move {
                let _ = audit::log_auth_attempt(&pool, Some(user_id), "login_failed", None, false).await;
            }
        });
        tracing::warn!("Failed login attempt for user: {}", user.username);
        return Err(AppError::Unauthorized("Invalid username or password".to_string()));
    }

    // Update last login timestamp
    user::update_last_login(db.pool(), user.id)
        .await
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    // Log successful authentication
    let _ = tokio::spawn({
        let pool = db.pool().clone();
        let user_id = user.id;
        async move {
            let _ = audit::log_auth_attempt(&pool, Some(user_id), "login_success", None, true).await;
        }
    });

    // Generate JWT token
    let token = generate_token(&user)
        .map_err(|e| AppError::InternalServerError(e.to_string()))?;

    tracing::info!("User {} logged in successfully", user.username);

    Ok(Json(LoginResponse {
        token,
        user: UserInfo::from(user),
    }))
}

/// Logout handler
/// 
/// Currently a placeholder - JWT tokens are stateless, so logout is handled client-side
/// In a production system, you might implement token blacklisting
async fn logout(auth_user: AuthUser) -> Json<serde_json::Value> {
    tracing::info!("User {} logged out", auth_user.username);
    
    Json(json!({
        "message": "Logged out successfully"
    }))
}

/// Verify token handler
/// 
/// Verifies that the JWT token is valid and returns user information
async fn verify(auth_user: AuthUser) -> Json<serde_json::Value> {
    Json(json!({
        "valid": true,
        "user": {
            "id": auth_user.user_id,
            "username": auth_user.username,
            "role": auth_user.role,
        }
    }))
}
