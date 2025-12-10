from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from .rag_pipeline import generate_treatment_advice


class SolutionRequest(BaseModel):
    cnn_label: str = Field(..., description="Label de la maladie prédite par le CNN")
    mode: str = Field(..., description="Mode de conduite : 'bio' ou 'conventionnel'")
    severity: str = Field(..., description="Niveau de gravité : 'faible', 'moderee' ou 'forte'")
    area_m2: float = Field(..., gt=0, description="Surface concernée en m²")
    date_iso: Optional[str] = Field(
        None,
        description="Date ISO (YYYY-MM-DD) utilisée pour estimer la saison",
    )
    location: Optional[str] = Field(
        None,
        description="Localisation texte (facultatif, pour info)",
    )


class SolutionResponse(BaseModel):
    data: Dict[str, Any]


# 🔴 C’est CETTE variable que Uvicorn cherche : `app`
app = FastAPI(
    title="VitiScan Solutions API",
    description="API de recommandations de traitements basée sur un pipeline RAG.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/solutions", response_model=SolutionResponse)
def get_solutions(request: SolutionRequest):
    """
    Endpoint principal : prend une prédiction de maladie + contexte
    et renvoie un plan de traitement structuré.
    """
    advice = generate_treatment_advice(request.dict())
    return {"data": advice}
