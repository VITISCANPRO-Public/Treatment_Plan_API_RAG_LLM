import os
from dotenv import load_dotenv
import weaviate

load_dotenv()


def get_weaviate_client() -> weaviate.WeaviateClient | None:
    """
    Essaie de se connecter à une instance Weaviate locale.
    - Si tout va bien : on retourne un client.
    - En cas d'erreur (version incompatible, pas démarré, etc.) : on retourne None
      et l'API continuera à fonctionner en mode "sans RAG".
    """

    try:
        # En local, avec Docker (ports par défaut), c'est suffisant.
        client = weaviate.connect_to_local()
        return client
    except Exception as e:
        # Log simple côté backend (pour toi)
        print(f"[RAG] Weaviate indisponible ou incompatible : {e}")
        return None


def search_treatment_chunks(
    client: weaviate.WeaviateClient | None,
    cnn_label: str,
    mode: str,
    severity: str,
    top_k: int = 5,
):
    """
    Squelette de requête vers Weaviate.
    - Si le client est None → on renvoie juste une liste vide.
    - Plus tard on ajoutera la vraie requête vectorielle.
    """

    if client is None:
        return []

    # TODO: implémenter la vraie requête vers Weaviate.
    return []
