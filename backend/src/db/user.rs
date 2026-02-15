use anyhow::{Context, Result};
use sqlx::PgPool;
use uuid::Uuid;
use chrono::Utc;

use crate::models::{User, UserRole};

/// Create a new user in the database
pub async fn create_user(
    pool: &PgPool,
    username: &str,
    password_hash: &str,
    email: Option<&str>,
    role: UserRole,
) -> Result<User> {
    let user = sqlx::query_as::<_, User>(
        r#"
        INSERT INTO users (username, password_hash, email, role)
        VALUES ($1, $2, $3, $4)
        RETURNING id, username, password_hash, email, role, created_at, last_login
        "#,
    )
    .bind(username)
    .bind(password_hash)
    .bind(email)
    .bind(role)
    .fetch_one(pool)
    .await
    .context("Failed to create user")?;

    Ok(user)
}

/// Find a user by username
pub async fn find_by_username(pool: &PgPool, username: &str) -> Result<Option<User>> {
    let user = sqlx::query_as::<_, User>(
        r#"
        SELECT id, username, password_hash, email, role, created_at, last_login
        FROM users
        WHERE username = $1
        "#,
    )
    .bind(username)
    .fetch_optional(pool)
    .await
    .context("Failed to find user by username")?;

    Ok(user)
}

/// Find a user by ID
pub async fn find_by_id(pool: &PgPool, user_id: Uuid) -> Result<Option<User>> {
    let user = sqlx::query_as::<_, User>(
        r#"
        SELECT id, username, password_hash, email, role, created_at, last_login
        FROM users
        WHERE id = $1
        "#,
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await
    .context("Failed to find user by ID")?;

    Ok(user)
}

/// Update user's last login timestamp
pub async fn update_last_login(pool: &PgPool, user_id: Uuid) -> Result<()> {
    sqlx::query(
        r#"
        UPDATE users
        SET last_login = $1
        WHERE id = $2
        "#,
    )
    .bind(Utc::now())
    .bind(user_id)
    .execute(pool)
    .await
    .context("Failed to update last login")?;

    Ok(())
}

/// Check if a username already exists
pub async fn username_exists(pool: &PgPool, username: &str) -> Result<bool> {
    let exists: (bool,) = sqlx::query_as(
        r#"
        SELECT EXISTS(SELECT 1 FROM users WHERE username = $1)
        "#,
    )
    .bind(username)
    .fetch_one(pool)
    .await
    .context("Failed to check username existence")?;

    Ok(exists.0)
}
