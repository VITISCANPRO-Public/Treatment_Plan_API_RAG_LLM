"""
test_api_integration.py — Integration tests for the Treatment Plan API endpoints.

Strategy:
  The RAG pipeline (Weaviate + HuggingFace LLM) is fully mocked.
  We test the API structure: routing, validation, response format.
  We do NOT test that the LLM produces good recommendations —
  that is a quality concern, not a structural one.

  The mock patches app.main.generate_treatment_advice so the
  FastAPI endpoint receives a realistic response without any cloud call.

Endpoints covered:
  GET  /          → health check
  GET  /health    → detailed health check
  POST /solutions → treatment plan generation
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


# ═══════════════════════════════════════════════════════════════════════════════
# Shared mock response — mimics exactly what generate_treatment_advice() returns
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_TREATMENT_RESPONSE = {
    "cnn_label":    "plasmopara_viticola",
    "disease_name": "Downy Mildew",
    "mode":         "conventional",
    "area_m2":      1000.0,
    "severity":     "moderate",
    "season":       "spring",
    "treatment_plan": {
        "area_m2":                       1000.0,
        "dose_l_ha":                     1.6,
        "volume_bouillie_l_ha":          250.0,
        "estimated_product_l_for_area":  0.176,
        "estimated_volume_l_for_area":   27.5,
        "configured":                    True,
    },
    "diagnostic":        "Downy mildew detected. Apply copper-based treatment immediately.",
    "treatment_actions": [
        "Apply copper hydroxide at 250 L/ha spray volume.",
        "Remove and destroy infected leaves.",
    ],
    "preventive_actions": [
        "Improve canopy ventilation through leaf removal.",
        "Monitor weather forecasts to anticipate future risk periods.",
    ],
    "warnings": [
        "These recommendations are indicative only.",
        "Always verify local regulations and product labels before application.",
        "Respect pre-harvest intervals for all applied products.",
    ],
    "raw_llm_output": "mock-llm-output",
}

# ── Valid request payload used across multiple tests ──────────────────────────
VALID_PAYLOAD = {
    "cnn_label": "plasmopara_viticola",
    "mode":      "conventional",
    "severity":  "moderate",
    "area_m2":   1000.0,
    "date_iso":  "2024-05-15",
    "location":  "Bordeaux, France",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Client fixture — mocks generate_treatment_advice for all tests in this module
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """
    Returns a TestClient with generate_treatment_advice permanently mocked.
    The mock is active for every test in this module — no real Weaviate or
    HuggingFace call will ever be made during the test session.
    """
    with patch(
        "app.main.generate_treatment_advice",
        side_effect=lambda payload: MOCK_TREATMENT_RESPONSE.copy(),
    ):  
        with TestClient(app) as c:
            yield c


# ═══════════════════════════════════════════════════════════════════════════════
# GET /  — root health check
# ═══════════════════════════════════════════════════════════════════════════════

class TestRootEndpoint:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200, (
            f"GET / returned {response.status_code} instead of 200"
        )

    def test_root_has_message_field(self, client):
        data = client.get("/").json()
        assert "message" in data, "GET / response missing 'message' field"

    def test_root_has_status_field(self, client):
        data = client.get("/").json()
        assert "status" in data, "GET / response missing 'status' field"

    def test_root_status_is_ok(self, client):
        data = client.get("/").json()
        assert data["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health  — detailed health check
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# POST /solutions — treatment plan generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolutionsEndpoint:

    def test_solutions_returns_200_with_valid_payload(self, client):
        response = client.post("/solutions", json=VALID_PAYLOAD)
        assert response.status_code == 200, (
            f"POST /solutions returned {response.status_code}.\n"
            f"Response: {response.text}"
        )

    def test_solutions_response_has_data_field(self, client):
        """The response must be wrapped in a 'data' key (SolutionResponse schema)."""
        response = client.post("/solutions", json=VALID_PAYLOAD)
        assert "data" in response.json(), "Response missing top-level 'data' field"

    def test_solutions_data_has_cnn_label(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "cnn_label" in data

    def test_solutions_data_has_disease_name(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "disease_name" in data
        assert isinstance(data["disease_name"], str)
        assert len(data["disease_name"]) > 0

    def test_solutions_data_has_diagnostic(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "diagnostic" in data
        assert isinstance(data["diagnostic"], str)

    def test_solutions_data_has_treatment_actions(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "treatment_actions" in data
        assert isinstance(data["treatment_actions"], list)

    def test_solutions_data_has_preventive_actions(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "preventive_actions" in data
        assert isinstance(data["preventive_actions"], list)

    def test_solutions_data_has_warnings(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_solutions_data_has_treatment_plan(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "treatment_plan" in data
        assert isinstance(data["treatment_plan"], dict)

    def test_solutions_data_has_season(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "season" in data

    def test_solutions_cnn_label_matches_input(self, client):
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert data["cnn_label"] == VALID_PAYLOAD["cnn_label"]

    def test_solutions_raw_llm_output_hidden_by_default(self, client):
        """
        raw_llm_output must NOT appear in the default response.
        It is only included when debug=true is passed as a query param.
        """
        data = client.post("/solutions", json=VALID_PAYLOAD).json()["data"]
        assert "raw_llm_output" not in data, (
            "raw_llm_output should be hidden unless ?debug=true is set"
        )

    def test_solutions_raw_llm_output_visible_with_debug(self, client):
        """When ?debug=true, raw_llm_output must appear in the response."""
        data = client.post("/solutions?debug=true", json=VALID_PAYLOAD).json()["data"]
        assert "raw_llm_output" in data


# ═══════════════════════════════════════════════════════════════════════════════
# POST /solutions — input validation (FastAPI / Pydantic)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolutionsValidation:

    def test_missing_cnn_label_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "cnn_label"}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 422

    def test_missing_mode_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "mode"}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 422

    def test_missing_severity_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "severity"}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 422

    def test_missing_area_m2_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "area_m2"}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 422

    def test_negative_area_returns_422(self, client):
        """area_m2 has ge=0 constraint in the schema — negative value must be rejected."""
        payload = {**VALID_PAYLOAD, "area_m2": -100.0}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 422

    def test_optional_date_iso_can_be_omitted(self, client):
        """date_iso is Optional — omitting it must not raise a 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "date_iso"}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 200

    def test_optional_location_can_be_omitted(self, client):
        """location is Optional — omitting it must not raise a 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "location"}
        response = client.post("/solutions", json=payload)
        assert response.status_code == 200

    def test_empty_body_returns_422(self, client):
        response = client.post("/solutions", json={})
        assert response.status_code == 422

    def test_all_7_disease_classes_are_accepted(self, client):
        """
        The API must accept all 7 INRAE disease labels without 422.
        (Validation of label values is done inside the pipeline, not by Pydantic)
        """
        diseases = [
            "colomerus_vitis", "elsinoe_ampelina", "erysiphe_necator",
            "guignardia_bidwellii", "healthy",
            "phaeomoniella_chlamydospora", "plasmopara_viticola",
        ]
        for disease in diseases:
            payload  = {**VALID_PAYLOAD, "cnn_label": disease}
            response = client.post("/solutions", json=payload)
            assert response.status_code == 200, (
                f"POST /solutions returned {response.status_code} for disease '{disease}'"
            )