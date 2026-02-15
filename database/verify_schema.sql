-- Verification script for database schema
-- Run this after migrations to verify everything was created correctly

\echo '========================================='
\echo 'Database Schema Verification'
\echo '========================================='
\echo ''

-- Check PostGIS version
\echo 'PostGIS Version:'
SELECT PostGIS_Version();
\echo ''

-- Check all tables exist
\echo 'Tables Created:'
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
\echo ''

-- Check spatial indexes
\echo 'Spatial Indexes (GIST):'
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexdef LIKE '%USING gist%'
ORDER BY tablename, indexname;
\echo ''

-- Check foreign key constraints
\echo 'Foreign Key Constraints:'
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name;
\echo ''

-- Check check constraints
\echo 'Check Constraints:'
SELECT
    tc.table_name,
    tc.constraint_name,
    cc.check_clause
FROM information_schema.table_constraints tc
JOIN information_schema.check_constraints cc
    ON tc.constraint_name = cc.constraint_name
WHERE tc.constraint_type = 'CHECK'
ORDER BY tc.table_name;
\echo ''

-- Check views
\echo 'Views Created:'
SELECT 
    table_name as view_name
FROM information_schema.views
WHERE table_schema = 'public'
ORDER BY table_name;
\echo ''

-- Check functions
\echo 'Custom Functions:'
SELECT 
    routine_name,
    routine_type,
    data_type as return_type
FROM information_schema.routines
WHERE routine_schema = 'public'
  AND routine_name NOT LIKE 'pg_%'
  AND routine_name NOT LIKE 'st_%'
ORDER BY routine_name;
\echo ''

-- Check triggers
\echo 'Triggers:'
SELECT 
    trigger_name,
    event_object_table as table_name,
    action_timing,
    event_manipulation
FROM information_schema.triggers
WHERE trigger_schema = 'public'
ORDER BY event_object_table, trigger_name;
\echo ''

-- Verify geometry columns
\echo 'Geometry Columns:'
SELECT 
    f_table_name as table_name,
    f_geometry_column as column_name,
    type as geometry_type,
    srid
FROM geometry_columns
ORDER BY f_table_name;
\echo ''

-- Check table sizes
\echo 'Table Sizes:'
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
\echo ''

-- Test validation function
\echo 'Testing validate_indian_boundaries function:'
\echo 'Valid coordinates (New Delhi - 77.2090, 28.6139):'
SELECT validate_indian_boundaries(ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)) as is_valid;
\echo 'Invalid coordinates (London - -0.1276, 51.5074):'
SELECT validate_indian_boundaries(ST_SetSRID(ST_MakePoint(-0.1276, 51.5074), 4326)) as is_valid;
\echo ''

\echo '========================================='
\echo 'Verification Complete'
\echo '========================================='
