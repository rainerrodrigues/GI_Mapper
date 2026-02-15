use anyhow::{Context, Result};
use sqlx::PgPool;
use uuid::Uuid;
use std::net::IpAddr;

/// Log an authentication attempt
pub async fn log_auth_attempt(
    pool: &PgPool,
    user_id: Option<Uuid>,
    action: &str,
    ip_address: Option<IpAddr>,
    success: bool,
) -> Result<()> {
    let details = serde_json::json!({
        "success": success,
    });

    sqlx::query(
        r#"
        INSERT INTO audit_log (user_id, action, entity_type, ip_address, details)
        VALUES ($1, $2, $3, $4, $5)
        "#,
    )
    .bind(user_id)
    .bind(action)
    .bind("authentication")
    .bind(ip_address.map(|ip| ip.to_string()))
    .bind(details)
    .execute(pool)
    .await
    .context("Failed to log authentication attempt")?;

    Ok(())
}

/// Log a general audit event
pub async fn log_audit_event(
    pool: &PgPool,
    user_id: Uuid,
    action: &str,
    entity_type: &str,
    entity_id: Option<Uuid>,
    ip_address: Option<IpAddr>,
    details: Option<serde_json::Value>,
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO audit_log (user_id, action, entity_type, entity_id, ip_address, details)
        VALUES ($1, $2, $3, $4, $5, $6)
        "#,
    )
    .bind(user_id)
    .bind(action)
    .bind(entity_type)
    .bind(entity_id)
    .bind(ip_address.map(|ip| ip.to_string()))
    .bind(details)
    .execute(pool)
    .await
    .context("Failed to log audit event")?;

    Ok(())
}
