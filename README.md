# VitiScan – RAG LLM & API Solutions

Ce dépôt contient la partie **RAG + API de recommandations de traitements** pour le projet VitiScan Pro.

Objectif :  
À partir d'un diagnostic de maladie de la vigne (fourni par le modèle CNN) et d'informations de contexte
(surface en m², gravité, mode bio / conventionnel, date), cette API propose :

- un plan de traitement synthétique et scientifique,
- des volumes de bouillie à préparer,
- des quantités de produit à utiliser,
- des mesures préventives et des rappels de sécurité.

## Architecture du dépôt

```text
vitiscan-rag-llm/
├─ data/
│  └─ knowledge/           # fiches maladies au format .md
├─ app/
│  ├─ __init__.py
│  ├─ main.py              # API FastAPI /solutions
│  ├─ rag_pipeline.py      # logique RAG (Weaviate + LLM)
│  ├─ weaviate_client.py   # client Weaviate (connexion, requêtes)
│  ├─ ingestion.py         # ingestion des fiches dans Weaviate
│  ├─ dosage_rules.py      # règles de calcul des volumes / doses
│  └─ prompts.py           # templates de prompts LLM
├─ notebooks/
│  └─ 01_dev_rag.ipynb     # essais exploratoires
├─ docker-compose.yml      # Weaviate local
├─ env_vitiscan_rag.yml    # définition de l'environnement conda
├─ requirements.txt
└─ README.md
