-- Script to create an initial administrator user
-- This should be run after the database schema is created
-- 
-- Default credentials:
--   Username: admin
--   Password: admin123 (CHANGE THIS IMMEDIATELY IN PRODUCTION!)
--
-- The password hash below is bcrypt hash of "admin123" with cost 12

-- Check if admin user already exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM users WHERE username = 'admin') THEN
        -- Create admin user
        -- Password: admin123 (bcrypt hash with cost 12)
        INSERT INTO users (username, password_hash, email, role)
        VALUES (
            'admin',
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLaEiUM2',
            'admin@example.com',
            'administrator'
        );
        
        RAISE NOTICE 'Admin user created successfully';
        RAISE NOTICE 'Username: admin';
        RAISE NOTICE 'Password: admin123';
        RAISE NOTICE 'IMPORTANT: Change this password immediately!';
    ELSE
        RAISE NOTICE 'Admin user already exists';
    END IF;
END $$;

-- Create additional sample users for testing (optional)
DO $$
BEGIN
    -- Analyst user
    IF NOT EXISTS (SELECT 1 FROM users WHERE username = 'analyst') THEN
        INSERT INTO users (username, password_hash, email, role)
        VALUES (
            'analyst',
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLaEiUM2',
            'analyst@example.com',
            'analyst'
        );
        RAISE NOTICE 'Analyst user created (username: analyst, password: admin123)';
    END IF;

    -- Viewer user
    IF NOT EXISTS (SELECT 1 FROM users WHERE username = 'viewer') THEN
        INSERT INTO users (username, password_hash, email, role)
        VALUES (
            'viewer',
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLaEiUM2',
            'viewer@example.com',
            'viewer'
        );
        RAISE NOTICE 'Viewer user created (username: viewer, password: admin123)';
    END IF;
END $$;

-- Display all users
SELECT 
    username,
    email,
    role,
    created_at,
    last_login
FROM users
ORDER BY role, username;
