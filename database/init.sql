-- Initialize PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify PostGIS installation
SELECT PostGIS_Version();

-- Run migrations
-- Migration 001: Create complete database schema
\i /docker-entrypoint-initdb.d/migrations/001_create_schema.sql
