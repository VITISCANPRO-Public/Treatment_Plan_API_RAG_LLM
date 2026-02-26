---
title: VITISCANPRO SOLUTION API
emoji: 📊
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
---

# Vitiscan — Treatment Plan API

RAG-based treatment recommendation API for grapevine diseases.
Part of the **Vitiscan MLOps pipeline**.

## Overview

Receives a disease prediction from the Diagnostic API and returns a structured
treatment plan by combining:
1. **Weaviate** — vector database storing technical disease knowledge sheets
2. **LLM (Llama 3)** — generates actionable recommendations from retrieved context
3. **Dosage rules** — computes precise product volumes based on area and severity

## Architecture

```
POST /solutions
      │
      ├── 1. Infer season from date
      ├── 2. Retrieve relevant chunks from Weaviate (RAG)
      ├── 3. Build prompt + call LLM (HuggingFace router)
      ├── 4. Compute dosage (disease rules)
      └── 5. Return structured treatment plan
```

> **Graceful degradation:** If Weaviate is unavailable (e.g. HuggingFace Spaces without
> `WEAVIATE_URL` configured), the API falls back to static disease-specific responses
> instead of returning HTTP 500 errors.

## Classes

| CNN Label | Disease |
|-----------|---------|
| `colomerus_vitis` | Erinose |
| `elsinoe_ampelina` | Anthracnose |
| `erysiphe_necator` | Powdery Mildew |
| `guignardia_bidwellii` | Black Rot |
| `phaeomoniella_chlamydospora` | Esca |
| `plasmopara_viticola` | Downy Mildew |
| `healthy` | Healthy |

## Project Structure

```
Treatment-Plan-API-RAG-LLM/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline (tests + HuggingFace deploy)
├── app/
│   ├── __init__.py
│   ├── config.py               # Environment variables and constants
│   ├── dosage_rules.py         # Dosage rules and treatment products by disease
│   ├── ingestion.py            # Loads knowledge .md files into Weaviate
│   ├── llm_client.py           # HuggingFace LLM API wrapper
│   ├── main.py                 # FastAPI application and endpoints
│   ├── prompts.py              # LLM prompt construction
│   ├── rag_pipeline.py         # Main RAG pipeline
│   └── weaviate_client.py      # Weaviate connection and vector search
├── data/
│   └── knowledge/              # Technical disease sheets (.md)
│       ├── Anthracnose_elsinoe_ampelina.md
│       ├── Black_rot_guignardia_bidwellii.md
│       ├── Downy_mildew_plasmopara_viticola.md
│       ├── Erinose_colomerus_vitis.md
│       ├── Esca_phaeomoniella_chlamydospora.md
│       ├── Healthy.md
│       └── Powdery_mildew_erysiphe_necator.md
├── scripts/
│   └── test_rag.py             # Manual RAG retrieval validation (requires Weaviate)
├── tests/
│   ├── __init__.py
│   ├── test_api_integration.py # Integration tests (endpoints, mocked RAG pipeline)
│   └── test_units.py           # Unit tests (pure functions, no external services)
├── .env.template               # Environment variable template
├── .gitignore
├── conftest.py                 # pytest root config (fixes PYTHONPATH)
├── docker-compose.yml          # Local Weaviate instance
├── Dockerfile                  # Production image (python:3.10-slim)
├── environment.yml             # Conda environment for local development
├── README.md
└── requirements.txt            # Pinned dependencies for Docker
```

## Quickstart

**1. Clone and install dependencies**
```bash
git clone https://github.com/VITISCANPRO-Public/Treatment_Plan_API_RAG_LLM.git
cd Treatment-Plan-API-RAG-LLM
pip install -r requirements.txt
```

**2. Configure your `.env`**
```bash
cp .env.template .env
# Fill in HF_TOKEN, WEAVIATE_URL, WEAVIATE_API_KEY
```

**3. Start Weaviate locally**
```bash
docker-compose up -d
```

**4. Ingest knowledge base**
```bash
python -m app.ingestion
```

**5. Run the API**
```bash
uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```

API docs available at `http://127.0.0.1:9000/docs`

**6. Validate RAG retrieval (optional)**
```bash
python scripts/test_rag.py
```

> This script requires a running Weaviate instance with data already ingested.
> It is a manual validation tool, not an automated pytest test.

## Running Tests

Tests run without any external service (Weaviate and HuggingFace are fully mocked).

```bash
# All tests
pytest tests/ -v

# Unit tests only (pure functions — fastest)
pytest tests/test_units.py -v

# Integration tests only (API endpoints)
pytest tests/test_api_integration.py -v
```

| File | What is tested |
|------|----------------|
| `test_units.py` | `infer_season_from_date`, `compute_dosage`, `_normalize_cnn_label`, `parse_llm_structured_response` |
| `test_api_integration.py` | `GET /`, `GET /health`, `POST /solutions` (structure, validation, debug flag) |

## CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

```
push to main / pull request
        │
        ├── Job 1: Unit Tests          → pytest tests/test_units.py
        │
        ├── Job 2: Integration Tests   → pytest tests/test_api_integration.py
        │          (only if Job 1 passes)
        │
        └── Job 3: Deploy              → git push to HuggingFace Spaces
                   (only if Jobs 1 + 2 pass AND push to main)
```

**Required GitHub configuration (Settings → Secrets and variables → Actions):**

| Type | Name | Value |
|------|------|-------|
| Secret | `HF_TOKEN` | Your HuggingFace API token |
| Variable | `HF_USERNAME` | Your HuggingFace username |
| Variable | `HF_SPACE_NAME` | Your HuggingFace Space name |

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health check |
| POST | `/solutions` | Generate treatment plan |

### POST /solutions — Request

```json
{
  "cnn_label": "plasmopara_viticola",
  "mode":      "conventional",
  "severity":  "moderate",
  "area_m2":   1000.0,
  "date_iso":  "2024-05-15",
  "location":  "Bordeaux, France"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cnn_label` | string | ✅ | INRAE disease label predicted by the CNN |
| `mode` | string | ✅ | `"conventional"` or `"organic"` |
| `severity` | string | ✅ | `"low"`, `"moderate"` or `"high"` |
| `area_m2` | float ≥ 0 | ✅ | Affected area in m² |
| `date_iso` | string | ❌ | ISO date (YYYY-MM-DD) to infer the season |
| `location` | string | ❌ | Text location (informational only) |

Add `?debug=true` to include the raw LLM output in the response.

## Configuration

All environment variables are defined in `app/config.py` and loaded via `.env`.

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (required for LLM calls) | — |
| `HF_MODEL_ID` | LLM model ID | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `HF_API_URL` | HuggingFace router URL | `https://router.huggingface.co/v1/chat/completions` |
| `WEAVIATE_URL` | Weaviate Cloud URL — leave empty for local | `""` |
| `WEAVIATE_API_KEY` | Weaviate Cloud API key | `""` |
| `DEBUG` | Set to `"true"` to enable verbose pipeline logging | `"false"` |

## Deployment

Deployed on HuggingFace Spaces (Docker) at:
`https://mouniat-vitiscanpro-solution-api.hf.space`

> For production, set `WEAVIATE_URL`, `WEAVIATE_API_KEY` and `HF_TOKEN`
> in HuggingFace Space secrets (Settings → Variables and secrets).

## Requirements

- Python 3.10
- Weaviate 1.27+
- Docker (for local Weaviate via `docker-compose`)
- See `requirements.txt` for the full pinned dependency list

## Author

**Mounia Tonazzini** — Agronomist Engineer & Data Scientist and Data Engineer

- HuggingFace: [huggingface.co/MouniaT](https://huggingface.co/MouniaT)
- LinkedIn: [www.linkedin.com/in/mounia-tonazzini](www.linkedin.com/in/mounia-tonazzini)
- GitHub: [github/Mounia-Agronomist-Datascientist](https://github.com/Mounia-Agronomist-Datascientist)
- Email : mounia.tonazzini@gmail.com