use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// User role for role-based access control
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "text")]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    #[sqlx(rename = "administrator")]
    Administrator,
    #[sqlx(rename = "analyst")]
    Analyst,
    #[sqlx(rename = "viewer")]
    Viewer,
}

impl UserRole {
    /// Check if this role has permission for a given operation
    pub fn has_permission(&self, required_role: &UserRole) -> bool {
        match (self, required_role) {
            // Administrator has all permissions
            (UserRole::Administrator, _) => true,
            // Analyst has analyst and viewer permissions
            (UserRole::Analyst, UserRole::Analyst) => true,
            (UserRole::Analyst, UserRole::Viewer) => true,
            // Viewer only has viewer permissions
            (UserRole::Viewer, UserRole::Viewer) => true,
            // All other combinations are denied
            _ => false,
        }
    }
}

/// User model from database
#[derive(Debug, Clone, FromRow, Serialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub email: Option<String>,
    pub role: UserRole,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

/// Request to create a new user
#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub password: String,
    pub email: Option<String>,
    pub role: UserRole,
}

/// Request to login
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

/// Response after successful login
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: UserInfo,
}

/// User information (without sensitive data)
#[derive(Debug, Serialize)]
pub struct UserInfo {
    pub id: Uuid,
    pub username: String,
    pub email: Option<String>,
    pub role: UserRole,
}

impl From<User> for UserInfo {
    fn from(user: User) -> Self {
        Self {
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role,
        }
    }
}

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,  // Subject (user ID)
    pub username: String,
    pub role: UserRole,
    pub exp: usize,   // Expiration time
    pub iat: usize,   // Issued at
}
