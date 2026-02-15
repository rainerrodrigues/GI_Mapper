.PHONY: help build up down logs clean test fmt db-shell db-verify db-reset

help:
	@echo "AI-Powered Blockchain GIS Platform - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs from all services"
	@echo "  make clean      - Remove all containers and volumes"
	@echo "  make test       - Run all tests"
	@echo "  make fmt        - Format code"
	@echo "  make db-shell   - Open PostgreSQL shell"
	@echo "  make db-verify  - Verify database schema"
	@echo "  make db-reset   - Reset database (WARNING: destroys all data)"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf target/
	rm -rf analytics/Manifest.toml

test:
	cd backend && cargo test
	cd analytics && julia --project=. -e 'using Pkg; Pkg.test()'

fmt:
	cd backend && cargo fmt
	cd analytics && julia --project=. -e 'using JuliaFormatter; format(".")'

db-shell:
	docker-compose exec postgres psql -U gis_user -d gis_platform

db-verify:
	@echo "Verifying database schema..."
	docker-compose exec -T postgres psql -U gis_user -d gis_platform -f /docker-entrypoint-initdb.d/verify_schema.sql

db-reset:
	@echo "WARNING: This will destroy all data in the database!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v postgres; \
		docker volume rm gis-platform_postgres_data 2>/dev/null || true; \
		docker-compose up -d postgres; \
		echo "Database reset complete. Waiting for initialization..."; \
		sleep 10; \
		make db-verify; \
	fi

backend-shell:
	docker-compose exec backend /bin/bash

analytics-shell:
	docker-compose exec analytics /bin/bash
