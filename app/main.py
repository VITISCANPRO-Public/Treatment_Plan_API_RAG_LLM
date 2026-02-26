"""
main.py — FastAPI application for the Vitiscan Treatment Plan API.
Receives a disease prediction and returns a structured treatment plan via RAG pipeline.
"""


from fastapi import FastAPI, Query

from app.rag_pipeline import generate_treatment_advice
from app.schemas import HealthResponse, SolutionRequest, SolutionResponse


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


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Detailed health check endpoint."""
    return {"status": "ok"}


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