# app/weaviate_client.py

import os
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from typing import Any
from dotenv import load_dotenv
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
from sentence_transformers import SentenceTransformer

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

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
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY") 

    if url:
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=auth,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=60)
            ),
        )
    else:
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


# ---------- Recherche de chunks de traitement ----------

def search_treatment_chunks(
    client: weaviate.WeaviateClient,
    disease_input: str,
    mode: Optional[str],
    severity: Optional[str],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Retrieval RAG robuste :
    - accepte disease_input au format "downy_mildew" ou "Grape_Downy_mildew_leaf"
    - filtre par (cnn_label == ...) OR (disease_id == ...)
    - filtre mode_conduite si fourni, sinon pas de filtre mode
    - fallback : si 0 résultat avec mode -> relance sans mode
    """

    try:
        collection = client.collections.get("VitiScanKnowledge")
    except Exception as e:
        print(f"[RAG] Collection VitiScanKnowledge introuvable: {e}")
        return []

    # Normalisation légère
    key = (disease_input or "").strip()
    if not key:
        return []

    # Heuristique : si on reçoit "Grape_..." => c'est plutôt cnn_label
    # sinon => c'est plutôt disease_id (ex: downy_mildew)
    cnn_label_value = key if key.startswith("Grape_") else None
    disease_id_value = None if key.startswith("Grape_") else key

    # Query text (sert à l'embedding)
    query_text = (
        f"Recommandations de traitement vigne pour {key}. "
        f"Mode: {mode or 'non spécifié'}. Gravité: {severity or 'non spécifiée'}. "
        "Inclure diagnostic, actions curatives, prévention, et précautions."
    )

    embedder = get_embedder()
    query_vector = embedder.encode(query_text).tolist()

    # Filtre maladie: cnn_label OU disease_id
    filters = []
    if cnn_label_value:
        filters.append(wvc.query.Filter.by_property("cnn_label").equal(cnn_label_value))
    if disease_id_value:
        filters.append(wvc.query.Filter.by_property("disease_id").equal(disease_id_value))

    if not filters:
        return []

    disease_filter = filters[0]
    for f in filters[1:]:
        disease_filter = disease_filter | f

    def run_query(with_mode: bool) -> List[Dict[str, Any]]:
        where_filter = disease_filter
        if with_mode and mode:
            mode_filter = wvc.query.Filter.by_property("mode_conduite").contains_any([mode])
            where_filter = where_filter & mode_filter

        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=where_filter,
                return_metadata=wvc.query.MetadataQuery(distance=True),
            )
        except Exception as e:
            print(f"[RAG] Erreur near_vector: {e}")
            return []

        chunks: List[Dict[str, Any]] = []
        for obj in response.objects:
            props = obj.properties or {}
            text = props.get("text", "")
            if not text:
                continue

            meta = getattr(obj, "metadata", None)
            distance = getattr(meta, "distance", None) if meta else None

            chunks.append(
                {
                    "text": text,
                    "section": props.get("section", ""),
                    "disease_id": props.get("disease_id", ""),
                    "cnn_label": props.get("cnn_label", ""),
                    "nom_fr": props.get("nom_fr", ""),
                    "mode_conduite": props.get("mode_conduite", None),
                    "distance": distance,
                }
            )

        return chunks

    # 1) Essai avec filtre mode
    chunks = run_query(with_mode=True)

    # 2) Fallback sans filtre mode si vide
    if not chunks:
        chunks = run_query(with_mode=False)

    return chunks