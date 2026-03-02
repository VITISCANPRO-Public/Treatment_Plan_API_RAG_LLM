"""
main.py — FastAPI application for the Vitiscan Treatment Plan API.
Receives a disease prediction and returns a structured treatment plan via RAG pipeline.
"""


from fastapi import FastAPI, Query

from app.rag_pipeline import generate_treatment_advice
from app.schemas import HealthResponse, SolutionRequest, SolutionResponse, DetailedHealthResponse
from app.weaviate_client import weaviate_available
from app.config import HF_TOKEN

# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="Vitiscan Treatment Plan API",
    description=(
        "RAG-based treatment recommendation API for grapevine diseases. "
        "Combines a Weaviate knowledge base with an LLM to generate structured treatment plans."
    ),
    version="1.0.0",
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def root():
    """Health check — confirms the API is running."""
    return {"message": "Vitiscan Treatment Plan API is running", "status": "ok"}


@app.get("/health", response_model=DetailedHealthResponse)
def health_check():
    """
    Detailed health check endpoint.
    
    Checks:
    - Weaviate availability (cloud or local)
    - HuggingFace token configuration
    
    Returns 'ok' if all components are available,
    'degraded' if running in fallback mode.
    """
    weaviate_ok = weaviate_available()
    llm_ok = bool(HF_TOKEN and HF_TOKEN.strip())
    
    components = {
        "weaviate": {
            "status": "ok" if weaviate_ok else "fallback",
            "message": "Connected" if weaviate_ok else "Unavailable — using static responses",
        },
        "llm": {
            "status": "ok" if llm_ok else "not_configured",
            "message": "HF_TOKEN configured" if llm_ok else "HF_TOKEN missing",
        },
    }
    
    # Overall status: ok if both are good, degraded otherwise
    overall = "ok" if (weaviate_ok and llm_ok) else "degraded"
    
    return {
        "status": overall,
        "components": components,
    }


@app.post("/solutions", response_model=SolutionResponse)
def get_solutions(
    request: SolutionRequest,
    debug: bool = Query(
        False,
        description="If true, includes raw LLM output in the response"
    ),
):
    """
    Main endpoint: receives a disease prediction + context
    and returns a structured treatment plan.

    - Retrieves relevant knowledge chunks from Weaviate
    - Builds a RAG prompt and calls the LLM
    - Computes dosage based on disease rules
    - Returns diagnosis, treatment actions, preventive measures and warnings
    """
    payload = request.model_dump()
    advice  = generate_treatment_advice(payload)

    if not debug:
        advice.pop("raw_llm_output", None)

    return {"data": advice}