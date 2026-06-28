# Industrial Operations Intelligence Platform

## Overview

An industry-agnostic data intelligence platform that converts raw business data into structured operational insights. Users can upload datasets from any industry, and the platform automatically profiles columns, suggests mappings, validates data, and organizes information into eight universal business domains.

## Universal Domains

- Assets
- Operations
- Quality
- Maintenance
- Inventory
- Workforce
- Finance
- Customers

## Architecture

### Extract
- Connectors read data from CSV files and external sources.

### Transform
- Profiling analyzes dataset structure.
- Mapping assigns columns to business domains.
- Validation ensures data quality.
- Transformations standardize data.

### Load
- Processed data is stored in the PostgreSQL Data Hub.

## Components

- **Ingestion Service** – ETL pipeline for onboarding and processing datasets.
- **API Gateway** – FastAPI service exposing platform APIs.
- **Data Hub** – PostgreSQL storage for metadata and business metrics.
- **Analytics Layer** – DuckDB views for fast reporting and KPI analysis.
- **Dashboard** – Plotly Dash user interface.
- **Data Simulator** – Generates sample datasets for testing and demos.

## Data Flow

```text
CSV Upload
    ↓
Profiling
    ↓
Mapping
    ↓
Validation
    ↓
Transformation
    ↓
Loading
    ↓
Analytics
    ↓
Dashboard
```

## Technology Stack

- Python
- FastAPI
- PostgreSQL
- DuckDB
- SQLAlchemy
- Plotly Dash
- Docker

## Key Features

- Industry-agnostic architecture
- Universal business domains
- Automated data onboarding
- Feature discovery and tracking
- Cross-domain analytics
- Dashboard-driven insights
- Dockerized deployment