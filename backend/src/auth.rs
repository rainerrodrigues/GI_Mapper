use anyhow::{Context, Result};
use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
    RequestPartsExt,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use bcrypt::{hash, verify, DEFAULT_COST};
use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use std::env;
use uuid::Uuid;

use crate::error::AppError;
use crate::models::{Claims, User, UserRole};

/// JWT secret key from environment
fn get_jwt_secret() -> String {
    env::var("JWT_SECRET").unwrap_or_else(|_| {
        tracing::warn!("JWT_SECRET not set, using default (INSECURE for production!)");
        "default_secret_key_change_in_production".to_string()
    })
}

/// JWT token expiration time in seconds (24 hours)
const TOKEN_EXPIRATION_SECONDS: i64 = 86400;

/// Hash a password using bcrypt
pub fn hash_password(password: &str) -> Result<String> {
    hash(password, DEFAULT_COST).context("Failed to hash password")
}

/// Verify a password against a hash
pub fn verify_password(password: &str, hash: &str) -> Result<bool> {
    verify(password, hash).context("Failed to verify password")
}

/// Generate a JWT token for a user
pub fn generate_token(user: &User) -> Result<String> {
    let now = Utc::now().timestamp() as usize;
    let exp = (Utc::now().timestamp() + TOKEN_EXPIRATION_SECONDS) as usize;

    let claims = Claims {
        sub: user.id.to_string(),
        username: user.username.clone(),
        role: user.role.clone(),
        exp,
        iat: now,
    };

    let secret = get_jwt_secret();
    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .context("Failed to generate JWT token")?;

    Ok(token)
}

/// Verify and decode a JWT token
pub fn verify_token(token: &str) -> Result<Claims> {
    let secret = get_jwt_secret();
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .context("Failed to verify JWT token")?;

    Ok(token_data.claims)
}

/// Extractor for authenticated user from JWT token
pub struct AuthUser {
    pub user_id: Uuid,
    pub username: String,
    pub role: UserRole,
}

#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Extract the Authorization header
        let TypedHeader(Authorization(bearer)) = parts
            .extract::<TypedHeader<Authorization<Bearer>>>()
            .await
            .map_err(|_| AppError::Unauthorized("Missing or invalid authorization header".to_string()))?;

        // Verify the token
        let claims = verify_token(bearer.token())
            .map_err(|e| AppError::Unauthorized(format!("Invalid token: {}", e)))?;

        // Parse user ID
        let user_id = Uuid::parse_str(&claims.sub)
            .map_err(|_| AppError::Unauthorized("Invalid user ID in token".to_string()))?;

        Ok(AuthUser {
            user_id,
            username: claims.username,
            role: claims.role,
        })
    }
}

/// Extractor for role-based authorization
pub struct RequireRole(pub UserRole);

impl RequireRole {
    /// Check if the authenticated user has the required role
    pub fn check(&self, user: &AuthUser) -> Result<(), AppError> {
        if user.role.has_permission(&self.0) {
            Ok(())
        } else {
            Err(AppError::Forbidden(format!(
                "Insufficient permissions. Required role: {:?}",
                self.0
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hashing() {
        let password = "test_password_123";
        let hash = hash_password(password).unwrap();
        
        assert!(verify_password(password, &hash).unwrap());
        assert!(!verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_role_permissions() {
        let admin = UserRole::Administrator;
        let analyst = UserRole::Analyst;
        let viewer = UserRole::Viewer;

        // Administrator has all permissions
        assert!(admin.has_permission(&UserRole::Administrator));
        assert!(admin.has_permission(&UserRole::Analyst));
        assert!(admin.has_permission(&UserRole::Viewer));

        // Analyst has analyst and viewer permissions
        assert!(!analyst.has_permission(&UserRole::Administrator));
        assert!(analyst.has_permission(&UserRole::Analyst));
        assert!(analyst.has_permission(&UserRole::Viewer));

        // Viewer only has viewer permissions
        assert!(!viewer.has_permission(&UserRole::Administrator));
        assert!(!viewer.has_permission(&UserRole::Analyst));
        assert!(viewer.has_permission(&UserRole::Viewer));
    }
}
