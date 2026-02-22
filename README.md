---
title: Vitiscan Treatment Plan API
emoji: 🍇
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
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

## Disease Classes (INRAE)

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
├── app/
│   ├── config.py           # Environment variables and constants
│   ├── dosage_rules.py     # Dosage rules and treatment products by disease
│   ├── ingestion.py        # Loads knowledge .md files into Weaviate
│   ├── llm_client.py       # HuggingFace LLM API wrapper
│   ├── main.py             # FastAPI application and endpoints
│   ├── prompts.py          # LLM prompt construction
│   ├── rag_pipeline.py     # Main RAG pipeline
│   └── weaviate_client.py  # Weaviate connection and vector search
├── data/
│   └── knowledge/          # Technical disease sheets (.md)
│       ├── colomerus_vitis.md
│       ├── elsinoe_ampelina.md
│       ├── erysiphe_necator.md
│       ├── guignardia_bidwellii.md
│       ├── healthy.md
│       ├── phaeomoniella_chlamydospora.md
│       └── plasmopara_viticola.md
├── docker-compose.yml      # Local Weaviate instance
├── test_rag.py             # Manual RAG retrieval test
├── Dockerfile
└── requirements.txt
```

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure your `.env`**
```bash
cp .env.example .env
# Fill in HF_API_TOKEN, WEAVIATE_URL, WEAVIATE_API_KEY
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

**6. Test RAG retrieval**
```bash
python test_rag.py
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health check |
| POST | `/solutions` | Generate treatment plan |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_API_TOKEN` | HuggingFace API token | — |
| `HF_MODEL_ID` | LLM model ID | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `HF_API_URL` | HuggingFace router URL | `https://router.huggingface.co/v1/chat/completions` |
| `WEAVIATE_URL` | Weaviate Cloud URL (empty = local) | `""` |
| `WEAVIATE_API_KEY` | Weaviate Cloud API key | `""` |

## Deployment

Deployed on HuggingFace Spaces (Docker) at:  
`https://mouniat-vitiscanpro-solution-api.hf.space`

> **Note:** Requires a running Weaviate instance.  
> For production, set `WEAVIATE_URL` and `WEAVIATE_API_KEY` in HuggingFace Secrets.

## Requirements

- Python 3.11
- Weaviate 1.27+
- Docker (for local Weaviate)
- See `requirements.txt` for full list