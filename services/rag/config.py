"""Configuration constants for the RAG pipeline.

Every tunable the document assistant needs -- chunk size, overlap, retrieval
depth, and answer length -- is read from the shared platform settings, so the
embedder, schema dimension, retriever, and QA chain can never disagree about
which model or dimension is in use.
"""
