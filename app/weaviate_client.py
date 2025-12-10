# app/weaviate_client.py

import os
from typing import List, Dict, Optional
from contextlib import contextmanager

from dotenv import load_dotenv
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------- Embedder global ----------

_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """
    Retourne un modèle SentenceTransformer (chargé une seule fois).
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


# ---------- Client Weaviate (context manager) ----------

@contextmanager
def weaviate_client():
    """
    Ouvre une connexion à Weaviate en local et la ferme proprement
    quand on sort du bloc `with weaviate_client() as client:`.
    """
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30)
        ),
    )
    try:
        yield client
    finally:
        client.close()


# ---------- Recherche de chunks de traitement ----------

def search_treatment_chunks(
    client: weaviate.WeaviateClient,
    cnn_label: str,
    mode: str,
    severity: str,
    top_k: int = 5,
) -> List[Dict[str, str]]:
    """
    Recherche dans la collection VitiScanKnowledge des chunks pertinents pour :
    - une maladie (cnn_label),
    - un mode de conduite (bio / conventionnel),
    - un niveau de gravité.

    Retourne une liste de dicts :
    [
      {
        "text": "...",
        "section": "...",
        "disease_id": "...",
        "nom_fr": "...",
      },
      ...
    ]
    """

    try:
        collection = client.collections.get("VitiScanKnowledge")
    except Exception as e:
        print(f"[RAG] Impossible de récupérer la collection VitiScanKnowledge : {e}")
        return []

    # Texte de requête
    query_text = (
        f"Recommandations de traitement pour la maladie de la vigne {cnn_label} "
        f"en mode {mode}, gravité {severity}. "
        "Inclure diagnostic, stratégie de traitement et mesures préventives."
    )

    embedder = get_embedder()
    query_vector = embedder.encode(query_text).tolist()

    # Filtre maladie + mode de conduite
    disease_filter = wvc.query.Filter.by_property("cnn_label").equal(cnn_label)
    mode_filter = wvc.query.Filter.by_property("mode_conduite").like(mode)
    where_filter = disease_filter & mode_filter

    try:
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=where_filter,
        )
    except Exception as e:
        print(f"[RAG] Erreur lors de la requête near_vector : {e}")
        return []

    chunks: List[Dict[str, str]] = []

    for obj in response.objects:
        props = obj.properties

        text = props.get("text", "")
        section = props.get("section", "")
        disease_id = props.get("disease_id", "")
        nom_fr = props.get("nom_fr", "")

        if not text:
            continue

        chunks.append(
            {
                "text": text,
                "section": section,
                "disease_id": disease_id,
                "nom_fr": nom_fr,
            }
        )

    return chunks
