"""
weaviate_client.py — Weaviate connection manager.

Connection strategy:
- Production (WEAVIATE_URL set)     : connects to Weaviate Cloud
- Local development (no HF env var) : connects to localhost:8080
- HuggingFace without WEAVIATE_URL  : yields None (graceful degradation)
                                      → rag_pipeline falls back to static responses
"""

import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import AdditionalConfig, Timeout

load_dotenv()

logger = logging.getLogger(__name__)

# ── Embedder (loaded once globally) ───────────────────────────────────────────

_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """
    Returns a SentenceTransformer model, loaded once and reused.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


# ── Deployment detection helpers ───────────────────────────────────────────────

def is_deployed() -> bool:
    """Returns True if running inside a deployed environment (HuggingFace, GCP, etc.)"""
    return bool(
        os.getenv("HF_SPACE_ID") or
        os.getenv("SPACE_ID") or
        os.getenv("K_SERVICE")
    )


def weaviate_available() -> bool:
    """
    Checks whether a Weaviate instance is reachable.

    Returns True if:
    - WEAVIATE_URL is configured (cloud instance), OR
    - We are running locally (not in a deployed environment)

    Returns False if deployed but WEAVIATE_URL is not set.
    In that case, rag_pipeline will use static fallback responses instead.
    """
    url = (os.getenv("WEAVIATE_URL") or "").strip()

    if url:
        return True       # cloud instance configured → available

    if is_deployed():
        logger.warning(
            "WEAVIATE_URL is not set in this deployed environment. "
            "Weaviate is unavailable — falling back to static responses. "
            "To enable full RAG functionality, set WEAVIATE_URL and "
            "WEAVIATE_API_KEY in HuggingFace Space secrets."
        )
        return False      # deployed without cloud instance → not available

    return True           # local development → assume localhost is running


# ── Weaviate client (context manager) ─────────────────────────────────────────

@contextmanager
def weaviate_client():
    """
    Context manager that opens and closes the Weaviate connection.

    Modes:
    - WEAVIATE_URL set              : connects to Weaviate Cloud
    - Local (no deployed env vars)  : connects to localhost:8080
    - Deployed without WEAVIATE_URL : yields None instead of raising RuntimeError
                                      → caller must check for None before using client

    Usage:
        with weaviate_client() as client:
            if client is None:
                return fallback_response()
            # ... use client normally
    """
    url     = (os.getenv("WEAVIATE_URL") or "").strip()
    api_key = (os.getenv("WEAVIATE_API_KEY") or "").strip()

    # ── HuggingFace without WEAVIATE_URL → yield None instead of crashing ─────
    # Previously this raised RuntimeError which caused HTTP 500 on all requests.
    # Now it yields None so the caller can fall back to static responses gracefully.
    if not url and is_deployed():
        logger.warning(
            "Deployed environment detected but WEAVIATE_URL is not set. "
            "Yielding None client — RAG pipeline will use static fallback responses."
        )
        yield None
        return

    # ── Cloud instance (Weaviate Cloud) ────────────────────────────────────────
    if url:
        logger.info(f"Connecting to Weaviate Cloud: {url}")
        auth   = weaviate.auth.AuthApiKey(api_key) if api_key else None
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=auth,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=60)
            ),
        )

    # ── Local instance (development only) ─────────────────────────────────────
    else:
        logger.info("Connecting to local Weaviate instance (localhost:8080)")
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=60)
            ),
        )

    try:
        yield client
    finally:
        client.close()
        logger.info("Weaviate connection closed.")


# ── RAG search ─────────────────────────────────────────────────────────────────

def search_treatment_chunks(
    client: weaviate.WeaviateClient,
    disease_input: str,
    mode: Optional[str],
    severity: Optional[str],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Robust RAG retrieval:
    - Accepts disease_input as INRAE scientific name (e.g. 'plasmopara_viticola')
    - Filters by (cnn_label == ...) OR (disease_id == ...)
    - Filters by farming_mode if provided, otherwise no mode filter
    - Fallback: if 0 results with mode filter → retries without mode filter

    Args:
        client:        Active Weaviate client
        disease_input: CNN label or disease ID
        mode:          Farming mode ('conventional' or 'organic')
        severity:      Severity level ('low', 'moderate', 'high')
        top_k:         Maximum number of chunks to return

    Returns:
        List of chunk dicts with text and metadata
    """
    try:
        collection = client.collections.get("VitiScanKnowledge")
    except Exception as e:
        logger.error(f"Collection VitiScanKnowledge not found: {e}")
        return []

    key = (disease_input or "").strip()
    if not key:
        return []

    # Build query text for embedding
    query_text = (
        f"Treatment recommendations for grapevine disease: {key}. "
        f"Farming mode: {mode or 'unspecified'}. Severity: {severity or 'unspecified'}. "
        "Include diagnosis, curative actions, prevention and safety precautions."
    )

    query_vector = get_embedder().encode(query_text).tolist()

    # Disease filter: match cnn_label OR disease_id
    disease_filter = (
        wvc.query.Filter.by_property("cnn_label").equal(key) |
        wvc.query.Filter.by_property("disease_id").equal(key)
    )

    def run_query(with_mode: bool) -> List[Dict[str, Any]]:
        where_filter = disease_filter
        if with_mode and mode:
            mode_filter  = wvc.query.Filter.by_property("farming_mode").contains_any([mode])
            where_filter = where_filter & mode_filter

        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=where_filter,
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )
        except Exception as e:
            logger.error(f"near_vector query error: {e}")
            return []

        chunks = []
        for obj in response.objects:
            props = obj.properties or {}
            text  = props.get("text", "")
            if not text:
                continue

            meta     = getattr(obj, "metadata", None)
            distance = getattr(meta, "distance", None) if meta else None

            chunks.append({
                "text":         text,
                "section":      props.get("section", ""),
                "disease_id":   props.get("disease_id", ""),
                "cnn_label":    props.get("cnn_label", ""),
                "disease_name": props.get("disease_name", ""),
                "farming_mode": props.get("farming_mode", None),
                "distance":     distance,
            })

        return chunks

    # First attempt: with farming_mode filter
    chunks = run_query(with_mode=True)

    # Fallback: without farming_mode filter if no results
    if not chunks:
        logger.warning(
            f"No results with mode filter '{mode}' for disease '{key}', "
            "retrying without mode filter..."
        )
        chunks = run_query(with_mode=False)

    return chunks