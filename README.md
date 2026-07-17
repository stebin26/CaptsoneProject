# Industrial Operations Intelligence Platform

An **industry-agnostic** operations intelligence platform. Any business uploads a
CSV; the platform profiles every column, maps it onto **eight universal business
domains**, and turns raw operational data into analytics, forecasts, anomaly
detection, risk scoring, cross-domain intelligence, document Q&A (RAG), and a
natural-language AI copilot — all behind one interface.

> **The thesis:** we don't model industries, we model the universal functions
> every operation shares. A new industry is onboarded through **config + a
> one-time confirmation**, not new code.

---

## The eight universal domains

`Assets` · `Operations` · `Quality` · `Maintenance` · `Inventory` ·
`Workforce` · `Finance` · `Customers`

Manufacturing, telecom, aerospace, education — different column names, same eight
domains. The ingestion layer is the only place industry specificity lives.

---

## Architecture (four phases, all complete)

| Phase | Capability |
|-------|-----------|
| **1 — Data Hub** | CSV upload → auto-profiling → LLM-assisted domain mapping → user confirm → Postgres + DuckDB hub |
| **2 — Data Engineering** | Apache Airflow orchestration + PySpark analytics (KPIs, trends, entity features) |
| **3 — Intelligence** | ML engine (forecasting, anomaly detection, risk scoring), cross-domain intelligence engine, RAG document assistant |
| **4 — Agentic AI** | LangGraph ReAct copilot — 12 tools across 5 groups, deterministic pre-filter, grounded answers with an evidence trail |

Plus an **Executive Dashboard** (single-call operational summary) and
**enterprise Auth/RBAC** (JWT + permission-based access control).

---

## Tech stack

- **Storage:** PostgreSQL (system of record, `pgvector` for embeddings) + DuckDB (fast analytics)
- **Orchestration / processing:** Apache Airflow (CeleryExecutor) + Apache Spark (PySpark)
- **Backend:** FastAPI (JWT auth, RBAC, versioned `/api/v1` routes)
- **Frontend:** Plotly Dash (design-system driven, app shell + nine pages)
- **AI:** local Ollama (`llama3.2:3b`) for the agent — offline, free, on-premise; RAG uses local embeddings
- **Everything runs in Docker Compose** — 14 containers, one command

---

## Quick start

**Prerequisites:** Docker Desktop. (Optional: Ollama running on the host with
`llama3.2:3b` pulled, for the AI copilot — the rest of the platform works without it.)

```bash
# 1. Bring up the whole stack. Data seeds itself automatically.
docker compose up -d --build

# 2. Create the first admin user (one time).
docker compose exec api python -m api_app.auth.bootstrap \
    --email admin@ops.com --password ChangeMe123 --name "Platform Admin"
```

First start takes a few minutes (image builds + a ~100s DuckDB attach on the API).
When the API health check is green, open the dashboard.

| Service | URL | Credentials |
|---------|-----|-------------|
| **Dashboard** | http://localhost:8050 | `admin@ops.com` / `ChangeMe123` |
| **API docs** | http://localhost:8000/docs | Bearer token from `/api/v1/auth/login` |
| **Airflow** | http://localhost:8080 | `admin` / `admin` |
| **Spark master** | http://localhost:8081 | — |

---

## The end-to-end flow