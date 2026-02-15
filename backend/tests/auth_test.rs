use backend::{
    auth::{hash_password, verify_password, generate_token, verify_token},
    db::{user, Database},
    models::{User, UserRole},
};
use uuid::Uuid;

#[test]
fn test_password_hashing() {
    let password = "secure_password_123";
    let hash = hash_password(password).expect("Failed to hash password");
    
    // Verify correct password
    assert!(verify_password(password, &hash).expect("Failed to verify password"));
    
    // Verify incorrect password
    assert!(!verify_password("wrong_password", &hash).expect("Failed to verify password"));
}

#[test]
fn test_jwt_token_generation_and_verification() {
    // Create a mock user
    let user = User {
        id: Uuid::new_v4(),
        username: "testuser".to_string(),
        password_hash: "dummy_hash".to_string(),
        email: Some("test@example.com".to_string()),
        role: UserRole::Analyst,
        created_at: chrono::Utc::now(),
        last_login: None,
    };

    // Generate token
    let token = generate_token(&user).expect("Failed to generate token");
    assert!(!token.is_empty());

    // Verify token
    let claims = verify_token(&token).expect("Failed to verify token");
    assert_eq!(claims.sub, user.id.to_string());
    assert_eq!(claims.username, user.username);
    assert_eq!(claims.role, user.role);
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

#[tokio::test]
#[ignore] // Requires running database
async fn test_create_and_find_user() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();
    
    // Create a test user
    let username = format!("testuser_{}", Uuid::new_v4());
    let password = "test_password_123";
    let password_hash = hash_password(password).unwrap();
    
    let created_user = user::create_user(
        db.pool(),
        &username,
        &password_hash,
        Some("test@example.com"),
        UserRole::Analyst,
    )
    .await
    .expect("Failed to create user");

    assert_eq!(created_user.username, username);
    assert_eq!(created_user.role, UserRole::Analyst);

    // Find user by username
    let found_user = user::find_by_username(db.pool(), &username)
        .await
        .expect("Failed to find user")
        .expect("User not found");

    assert_eq!(found_user.id, created_user.id);
    assert_eq!(found_user.username, username);

    // Find user by ID
    let found_by_id = user::find_by_id(db.pool(), created_user.id)
        .await
        .expect("Failed to find user by ID")
        .expect("User not found");

    assert_eq!(found_by_id.id, created_user.id);

    // Verify password
    assert!(verify_password(password, &found_user.password_hash).unwrap());
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_username_exists() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();
    
    // Create a test user
    let username = format!("testuser_{}", Uuid::new_v4());
    let password_hash = hash_password("test_password").unwrap();
    
    user::create_user(
        db.pool(),
        &username,
        &password_hash,
        None,
        UserRole::Viewer,
    )
    .await
    .expect("Failed to create user");

    // Check that username exists
    assert!(user::username_exists(db.pool(), &username)
        .await
        .expect("Failed to check username existence"));

    // Check that non-existent username doesn't exist
    assert!(!user::username_exists(db.pool(), "nonexistent_user_12345")
        .await
        .expect("Failed to check username existence"));
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_update_last_login() {
    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();
    
    // Create a test user
    let username = format!("testuser_{}", Uuid::new_v4());
    let password_hash = hash_password("test_password").unwrap();
    
    let created_user = user::create_user(
        db.pool(),
        &username,
        &password_hash,
        None,
        UserRole::Viewer,
    )
    .await
    .expect("Failed to create user");

    assert!(created_user.last_login.is_none());

    // Update last login
    user::update_last_login(db.pool(), created_user.id)
        .await
        .expect("Failed to update last login");

    // Verify last login was updated
    let updated_user = user::find_by_id(db.pool(), created_user.id)
        .await
        .expect("Failed to find user")
        .expect("User not found");

    assert!(updated_user.last_login.is_some());
}
