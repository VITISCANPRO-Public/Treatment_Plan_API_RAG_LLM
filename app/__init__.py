"""
app — Vitiscan Treatment Plan API package.

This package contains the FastAPI application for RAG-based
treatment recommendations for grapevine diseases.

Modules:
    config          Environment variables and constants
    dosage_rules    Dosage calculations and treatment products
    ingestion       Knowledge base indexing into Weaviate
    llm_client      HuggingFace LLM API wrapper
    main            FastAPI application and endpoints
    prompts         LLM prompt construction
    rag_pipeline    Main RAG pipeline orchestration
    schemas         Pydantic request/response models
    weaviate_client Weaviate connection and vector search
"""