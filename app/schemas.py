"""
schemas.py — Pydantic models for the Vitiscan Treatment Plan API.

Defines the exact shape of all request and response data.
FastAPI uses these models to:
  - Validate incoming request bodies automatically
  - Generate the interactive /docs (Swagger) documentation
  - Serialize responses with consistent field names and types
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
#  REQUEST SCHEMAS
# ──────────────────────────────────────────────

class SolutionRequest(BaseModel):
    """Body for POST /solutions — all context needed to generate a treatment plan."""

    cnn_label: str = Field(
        ...,
        min_length=1,
        description="Disease label predicted by the CNN (INRAE scientific name, e.g. 'guignardia_bidwellii')",
    )
    mode: str = Field(
        ...,
        pattern="^(conventional|organic)$",
        description="Farming mode: 'conventional' or 'organic'",
    )
    severity: str = Field(
        ...,
        pattern="^(low|moderate|high)$",
        description="Severity level: 'low', 'moderate' or 'high'",
    )
    area_m2: float = Field(
        ...,
        ge=0,
        description="Affected area in square meters",
    )
    date_iso: Optional[str] = Field(
        None,
        description="ISO date (YYYY-MM-DD) used to infer the season for treatment timing",
    )
    location: Optional[str] = Field(
        None,
        description="Text location of the vineyard (optional, for contextual information only)",
    )


# ──────────────────────────────────────────────
#  RESPONSE SCHEMAS
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for GET / and GET /health"""
    message: Optional[str] = Field(None, description="Human-readable status message")
    status: str = Field(..., description="API health status ('ok')")


class SolutionResponse(BaseModel):
    """Response for POST /solutions — wraps the full treatment advice payload."""
    data: Dict[str, Any] = Field(
        ...,
        description=(
            "Structured treatment plan containing: diagnosis details, "
            "recommended treatment actions, preventive measures, "
            "dosage calculations, and warnings"
        ),
    )


class ErrorResponse(BaseModel):
    """Standard error response for 4xx and 5xx errors."""
    detail: str = Field(..., description="Human-readable error message")