# Authentication and Authorization Implementation

This document describes the complete authentication and authorization system implemented for the AI-Powered Blockchain GIS Platform.

## Overview

The system implements JWT-based authentication with bcrypt password hashing and role-based access control (RBAC) as specified in Requirements 14.1-14.4.

## Architecture

### Components

1. **Authentication Module** (`src/auth.rs`)
   - Password hashing and verification using bcrypt
   - JWT token generation and validation
   - Token expiration management (24 hours)
   - User authentication extractor for Axum

2. **User Models** (`src/models/user.rs`)
   - User data structures
   - Role definitions (Administrator, Analyst, Viewer)
   - Login/response DTOs
   - JWT claims structure

3. **Database Operations** (`src/db/user.rs`)
   - User CRUD operations
   - Username lookup and validation
   - Last login timestamp tracking

4. **Authentication Routes** (`src/routes/auth.rs`)
   - POST /api/v1/auth/login - User authentication
   - POST /api/v1/auth/logout - User logout
   - POST /api/v1/auth/verify - Token verification

5. **Authorization Middleware** (`src/auth_middleware.rs`)
   - Role-based route protection
   - Permission checking
   - Request filtering by role

6. **Audit Logging** (`src/db/audit.rs`)
   - Authentication attempt logging
   - Security event tracking
   - Compliance with Requirement 14.6

## Requirements Mapping

### Requirement 14.1: Authentication via Username and Password
**Implementation:**
- POST /api/v1/auth/login endpoint accepts username and password
- Credentials validated against database
- Bcrypt used for secure password verification
- Failed attempts logged for security monitoring

**Files:**
- `src/routes/auth.rs` - Login endpoint
- `src/auth.rs` - Password verification
- `src/db/user.rs` - User lookup

### Requirement 14.2: JWT Token Issuance
**Implementation:**
- JWT tokens generated on successful authentication
- Tokens include user ID, username, and role
- 24-hour expiration time (86400 seconds)
- Tokens signed with secret key from environment

**Files:**
- `src/auth.rs` - `generate_token()` function
- `src/models/user.rs` - Claims structure

**Token Structure:**
```json
{
  "sub": "user-uuid",
  "username": "user@example.com",
  "role": "analyst",
  "iat": 1234567890,
  "exp": 1234654290
}
```

### Requirement 14.3: Role-Based Access Control
**Implementation:**
- Three roles: Administrator, Analyst, Viewer
- Hierarchical permissions (Admin > Analyst > Viewer)
- Role stored in database and JWT token
- Middleware enforces role requirements

**Files:**
- `src/models/user.rs` - UserRole enum with `has_permission()` method
- `src/auth_middleware.rs` - Role checking middleware

**Permission Matrix:**
| Role | Administrator | Analyst | Viewer |
|------|--------------|---------|--------|
| Administrator | ✓ | ✓ | ✓ |
| Analyst | ✗ | ✓ | ✓ |
| Viewer | ✗ | ✗ | ✓ |

### Requirement 14.4: Permission Verification
**Implementation:**
- Middleware validates permissions before route execution
- AuthUser extractor provides user context to handlers
- Unauthorized access returns 403 Forbidden
- Permission checks logged for audit

**Files:**
- `src/auth_middleware.rs` - Permission checking middleware
- `src/auth.rs` - AuthUser extractor

**Usage Example:**
```rust
Router::new()
    .route("/data", get(read_data))
    .route_layer(middleware::from_fn(require_viewer))
    .route("/data", post(create_data))
    .route_layer(middleware::from_fn(require_analyst))
```

### Requirement 14.6: Authentication Logging
**Implementation:**
- All login attempts logged to audit_log table
- Includes timestamp, user ID, IP address, success/failure
- Failed attempts tracked for security monitoring
- Async logging to avoid blocking requests

**Files:**
- `src/db/audit.rs` - Audit logging functions
- `src/routes/auth.rs` - Login endpoint with logging

## Security Features

### Password Security
- **Bcrypt Hashing**: Cost factor 12 (2^12 iterations)
- **Salt**: Automatically generated per password
- **No Plain Text**: Passwords never stored or logged in plain text
- **Constant-Time Comparison**: Prevents timing attacks

### Token Security
- **Signed Tokens**: HMAC-SHA256 signature
- **Expiration**: 24-hour lifetime
- **Secret Key**: Configurable via JWT_SECRET environment variable
- **Validation**: Signature and expiration checked on every request

### Authorization Security
- **Least Privilege**: Users granted minimum necessary permissions
- **Hierarchical Roles**: Clear permission inheritance
- **Middleware Enforcement**: Authorization checked before route execution
- **Audit Trail**: All access attempts logged

## API Usage

### Login
```bash
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "admin",
    "email": "admin@example.com",
    "role": "administrator"
  }
}
```

### Authenticated Request
```bash
curl http://localhost:8080/api/v1/gi-products \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Verify Token
```bash
curl -X POST http://localhost:8080/api/v1/auth/verify \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Logout
```bash
curl -X POST http://localhost:8080/api/v1/auth/logout \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    role VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    CONSTRAINT chk_role CHECK (role IN ('administrator', 'analyst', 'viewer'))
);
```

### Audit Log Table
```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100),
    entity_type VARCHAR(50),
    entity_id UUID,
    ip_address INET,
    timestamp TIMESTAMP DEFAULT NOW(),
    details JSONB
);
```

## Initial Setup

### 1. Create Admin User

Run the SQL script to create initial users:

```bash
# Using PowerShell
.\scripts\create-admin-user.ps1

# Or manually with Docker
docker exec -i gis-platform-postgres psql -U gis_user -d gis_platform -f /scripts/create-admin-user.sql
```

Default credentials:
- **Admin**: username=`admin`, password=`admin123`
- **Analyst**: username=`analyst`, password=`admin123`
- **Viewer**: username=`viewer`, password=`admin123`

⚠️ **IMPORTANT**: Change these passwords immediately in production!

### 2. Configure JWT Secret

Set the JWT_SECRET environment variable:

```bash
# Development
export JWT_SECRET=your-secret-key-here

# Production (use a strong random key)
export JWT_SECRET=$(openssl rand -base64 32)
```

Or in `.env` file:
```
JWT_SECRET=your-secret-key-here
```

## Testing

### Unit Tests
```bash
cd backend
cargo test test_password_hashing
cargo test test_jwt_token_generation_and_verification
cargo test test_role_permissions
```

### Integration Tests (requires database)
```bash
cd backend
cargo test test_create_and_find_user -- --ignored
cargo test test_username_exists -- --ignored
cargo test test_update_last_login -- --ignored
```

## Error Handling

### Authentication Errors

| Error | Status Code | Description |
|-------|-------------|-------------|
| Invalid credentials | 401 Unauthorized | Username or password incorrect |
| Missing token | 401 Unauthorized | Authorization header missing |
| Invalid token | 401 Unauthorized | Token signature invalid or expired |
| Insufficient permissions | 403 Forbidden | User role lacks required permissions |

### Example Error Response
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid username or password"
  }
}
```

## Future Enhancements

### Account Lockout (Requirement 14.5)
- Track failed login attempts per user
- Lock account after 5 failed attempts
- 15-minute lockout period
- Admin override capability

### OAuth Support
- Google OAuth integration
- GitHub OAuth integration
- Configurable OAuth providers

### Token Refresh
- Refresh tokens for extended sessions
- Automatic token renewal
- Revocation support

### Multi-Factor Authentication
- TOTP-based 2FA
- SMS verification
- Backup codes

### Advanced Audit Features
- IP-based anomaly detection
- Geographic login tracking
- Session management
- Device fingerprinting

## Troubleshooting

### "JWT_SECRET not set" Warning
Set the JWT_SECRET environment variable. The system will use a default value in development, but this is insecure for production.

### "Invalid token" Error
- Check token expiration (24 hours)
- Verify JWT_SECRET matches between token generation and validation
- Ensure Authorization header format: `Bearer <token>`

### "Insufficient permissions" Error
- Verify user role in database
- Check route middleware configuration
- Confirm role hierarchy (Admin > Analyst > Viewer)

### Database Connection Issues
- Verify DATABASE_URL environment variable
- Check PostgreSQL is running
- Confirm users table exists (run migrations)

## References

- [JWT.io](https://jwt.io/) - JWT token debugger
- [Bcrypt](https://en.wikipedia.org/wiki/Bcrypt) - Password hashing algorithm
- [Axum Documentation](https://docs.rs/axum/) - Web framework
- [SQLx Documentation](https://docs.rs/sqlx/) - Database toolkit
