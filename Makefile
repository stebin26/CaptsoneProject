.PHONY: help up down restart build rebuild logs logs-api logs-dash ps \
        seed clean reset api-shell db-shell health

COMPOSE := docker compose

help:
	@echo "Operations Intelligence Platform — available commands:"
	@echo ""
	@echo "  make up         Build (if needed) and start the full stack"
	@echo "  make down       Stop and remove containers"
	@echo "  make restart    Restart the stack"
	@echo "  make build      Build all images"
	@echo "  make rebuild    Rebuild images from scratch (no cache)"
	@echo "  make logs       Tail logs from all services"
	@echo "  make logs-api   Tail API logs only"
	@echo "  make logs-dash  Tail dashboard logs only"
	@echo "  make ps         Show running services"
	@echo "  make seed       Re-run the data simulator seed"
	@echo "  make health     Check API health"
	@echo "  make api-shell  Open a shell in the API container"
	@echo "  make db-shell   Open a psql shell in Postgres"
	@echo "  make clean      Stop containers and remove volumes (DELETES DATA)"
	@echo "  make reset      Clean everything and rebuild from scratch"
	@echo ""

up:
	$(COMPOSE) up -d --build
	@echo ""
	@echo "Dashboard:  http://localhost:8050"
	@echo "API docs:   http://localhost:8000/docs"

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

build:
	$(COMPOSE) build

rebuild:
	$(COMPOSE) build --no-cache

logs:
	$(COMPOSE) logs -f

logs-api:
	$(COMPOSE) logs -f api

logs-dash:
	$(COMPOSE) logs -f dashboard

ps:
	$(COMPOSE) ps

seed:
	$(COMPOSE) run --rm seed python -m app.cli seed

health:
	@curl -s http://localhost:8000/health || echo "API not reachable"
	@echo ""

api-shell:
	$(COMPOSE) exec api /bin/bash

db-shell:
	$(COMPOSE) exec postgres psql -U $${OPS_POSTGRES_USER:-ops} -d $${OPS_POSTGRES_DB:-ops}

clean:
	$(COMPOSE) down -v

reset: clean
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d
	@echo ""
	@echo "Fresh stack running at http://localhost:8050"