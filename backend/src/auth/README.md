# Authentication and Authorization System

This module implements a complete authentication and authorization system for the AI-Powered Blockchain GIS Platform.

## Features

### 1. JWT Token-Based Authentication (Requirement 14.1, 14.2)

- **Password Hashing**: Uses bcrypt with default cost factor (12) for secure password storage
- **JWT Tokens**: Issues JSON Web Tokens with 24-hour expiration
- **Token Claims**: Include user ID, username, role, issued-at time, and expiration

### 2. Role-Based Access Control (Requirement 14.3, 14.4)

Three user roles with hierarchical permissions:

- **Administrator**: Full access to all operations
- **Analyst**: Can perform analysis operations and view data (includes Viewer permissions)
- **Viewer**: Read-only access to data

### 3. Authentication Endpoints

#### POST /api/v1/auth/login
Authenticate with username and password, receive JWT token.

**Request:**
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "user@example.com",
    "email": "user@example.com",
    "role": "analyst"
  }
}
```

#### POST /api/v1/auth/logout
Logout (client-side token removal, server logs the event).

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

#### POST /api/v1/auth/verify
Verify JWT token validity and get user information.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "valid": true,
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "user@example.com",
    "role": "analyst"
  }
}
```

### 4. Protected Routes

Routes can be protected using middleware:

```rust
use crate::auth_middleware::{require_admin, require_analyst, require_viewer};

Router::new()
    // Read operations - require viewer role or higher
    .route("/data", get(read_data))
    .route_layer(middleware::from_fn(require_viewer))
    
    // Write operations - require analyst role or higher
    .route("/data", post(create_data))
    .route_layer(middleware::from_fn(require_analyst))
    
    // Admin operations - require administrator role
    .route("/admin", post(admin_operation))
    .route_layer(middleware::from_fn(require_admin))
```

### 5. Extracting Authenticated User

Use the `AuthUser` extractor in route handlers:

```rust
async fn my_handler(auth_user: AuthUser) -> Result<Json<Response>, AppError> {
    // Access user information
    let user_id = auth_user.user_id;
    let username = auth_user.username;
    let role = auth_user.role;
    
    // Your logic here
}
```

### 6. Audit Logging (Requirement 14.6)

All authentication attempts are logged to the `audit_log` table with:
- User ID (if available)
- Action (login_success, login_failed, etc.)
- Timestamp
- IP address (when available)
- Success/failure status

## Security Features

1. **Password Security**
   - Bcrypt hashing with cost factor 12
   - Passwords never stored in plain text
   - Password verification uses constant-time comparison

2. **Token Security**
   - JWT tokens signed with secret key
   - 24-hour expiration
   - Tokens include role information for authorization
   - Secret key configurable via JWT_SECRET environment variable

3. **Authorization**
   - Role-based access control enforced at route level
   - Hierarchical permissions (Administrator > Analyst > Viewer)
   - Middleware validates permissions before route execution

4. **Audit Trail**
   - All authentication attempts logged
   - Failed login attempts tracked
   - Timestamps and IP addresses recorded

## Environment Variables

```bash
# JWT secret key (REQUIRED in production)
JWT_SECRET=your_secure_secret_key_here

# Database connection
DATABASE_URL=postgresql://user:password@localhost/database
```

## Usage Example

### Creating a User

```rust
use backend::auth::hash_password;
use backend::db::user;
use backend::models::UserRole;

let password_hash = hash_password("secure_password")?;

let user = user::create_user(
    pool,
    "username",
    &password_hash,
    Some("email@example.com"),
    UserRole::Analyst,
).await?;
```

### Authenticating

```bash
# Login
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use token in subsequent requests
curl http://localhost:8080/api/v1/gi-products \
  -H "Authorization: Bearer <token>"
```

## Testing

Run tests with:

```bash
# Unit tests (no database required)
cargo test test_password_hashing
cargo test test_jwt_token_generation_and_verification
cargo test test_role_permissions

# Integration tests (requires database)
cargo test test_create_and_find_user -- --ignored
cargo test test_username_exists -- --ignored
cargo test test_update_last_login -- --ignored
```

## Future Enhancements

1. **Account Lockout** (Requirement 14.5)
   - Track failed login attempts
   - Lock account after 5 failed attempts for 15 minutes
   - Implement in future iteration

2. **Token Refresh**
   - Implement refresh tokens for extended sessions
   - Allow token renewal without re-authentication

3. **OAuth Integration** (Requirement 14.1)
   - Support OAuth providers (Google, GitHub, etc.)
   - Implement OAuth flow alongside username/password

4. **Token Blacklisting**
   - Implement server-side token revocation
   - Use Redis for blacklist storage
   - Enable true logout functionality

5. **Multi-Factor Authentication**
   - Add TOTP-based 2FA
   - SMS verification option
   - Backup codes for account recovery
