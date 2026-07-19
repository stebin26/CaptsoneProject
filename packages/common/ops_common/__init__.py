"""Shared foundation package for every service in the platform.

Holds the pieces each service must agree on: configuration, database access,
structured logging, and the universal domain model. Nothing here is
service-specific, which is what lets the API, ingestion, ML, RAG, agent, and
dashboard layers all speak the same language.
"""
