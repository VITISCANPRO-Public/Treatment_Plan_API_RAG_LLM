import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import frontmatter
import weaviate
import weaviate.classes as wvc

from app.weaviate_client import weaviate_client, get_embedder

# Nom de la collection dans Weaviate
COLLECTION_NAME = "VitiScanKnowledge"


def load_markdown_files(knowledge_dir: Path) -> List[Dict[str, Any]]:
    """
    Charge tous les fichiers .md du dossier data/knowledge
    et retourne une liste de dicts {path, meta, content}.
    """
    md_files = sorted(knowledge_dir.glob("*.md"))
    fiches: List[Dict[str, Any]] = []

    for md_path in md_files:
        post = frontmatter.load(md_path)
        fiches.append(
            {
                "path": str(md_path),
                "meta": dict(post.metadata),
                "content": post.content,
            }
        )

    return fiches


def split_markdown_sections(content: str) -> List[Dict[str, str]]:
    """
    Découpe le contenu markdown en sections à partir des titres de niveau 1 '# '.

    Retourne une liste de:
    {
        "section_title": "1. Description et symptômes",
        "text": "Texte de la section..."
    }
    """
    lines = content.splitlines()
    sections: List[Dict[str, str]] = []

    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        heading_match = re.match(r"^#\s+(.*)", line.strip())
        if heading_match:
            # Sauvegarder la section précédente
            if current_title is not None and current_lines:
                sections.append(
                    {
                        "section_title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            # Nouvelle section
            current_title = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Dernière section
    if current_title is not None and current_lines:
        sections.append(
            {
                "section_title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections


def ensure_collection(client: weaviate.WeaviateClient):
    """
    Crée la collection VitiScanKnowledge si elle n'existe pas déjà.
    Vectors: self_provided (on envoie les embeddings nous-mêmes).
    """
    collections = client.collections
    try:
        coll = collections.get(COLLECTION_NAME)
        _ = coll.config.get()
        return coll
    except Exception:
        coll = collections.create(
            name=COLLECTION_NAME,
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="section",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="disease_id",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="cnn_label",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="nom_fr",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="type",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="categorie",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="mode_conduite",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )
        return coll


def build_chunk_objects(fiches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforme les fiches markdown en une liste de chunks prêts à être indexés.

    Chaque chunk contient:
    - text: texte complet du chunk (titre de section + contenu)
    - section: nom de la section
    - disease_id, cnn_label, nom_fr, type, categorie, mode_conduite: depuis le front matter
    """
    all_chunks: List[Dict[str, Any]] = []

    for fiche in fiches:
        meta = fiche["meta"]
        content = fiche["content"]
        sections = split_markdown_sections(content)

        disease_id = meta.get("id")
        cnn_label = meta.get("cnn_label")
        nom_fr = meta.get("nom_fr")
        disease_type = meta.get("type")
        categorie = meta.get("categorie")
        mode_conduite = meta.get("mode_conduite") or []

        # On stocke mode_conduite en chaîne simple pour simplifier les filtres
        mode_conduite_str = ", ".join(mode_conduite)

        for section in sections:
            section_title = section["section_title"]
            text_body = section["text"]

            full_text = f"{section_title}\n\n{text_body}".strip()

            chunk = {
                "text": full_text,
                "section": section_title,
                "disease_id": disease_id,
                "cnn_label": cnn_label,
                "nom_fr": nom_fr,
                "type": disease_type,
                "categorie": categorie,
                "mode_conduite": mode_conduite_str,
            }
            all_chunks.append(chunk)

    return all_chunks


def ingest_chunks_into_weaviate(chunks: List[Dict[str, Any]]):
    """
    Envoie tous les chunks dans Weaviate avec des embeddings SentenceTransformer.
    Utilise le context manager weaviate_client() pour ouvrir/fermer proprement la connexion.
    """

    # On ouvre la connexion Weaviate une fois pour toute la fonction
    with weaviate_client() as client:
        collection = ensure_collection(client)

        print(f"[INGESTION] Nombre de chunks à indexer: {len(chunks)}")

        # On réutilise le même embedder que dans la partie search
        embedder = get_embedder()

        with collection.batch.dynamic() as batch:
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk["text"]
                vector = embedder.encode(text)
                vector_list = vector.tolist()

                batch.add_object(
                    properties={
                        "text": chunk["text"],
                        "section": chunk["section"],
                        "disease_id": chunk["disease_id"],
                        "cnn_label": chunk["cnn_label"],
                        "nom_fr": chunk["nom_fr"],
                        "type": chunk["type"],
                        "categorie": chunk["categorie"],
                        "mode_conduite": chunk["mode_conduite"],
                    },
                    vector=vector_list,
                )

                if idx % 20 == 0:
                    print(f"[INGESTION] {idx} chunks envoyés...")

            if batch.number_errors > 0:
                print(f"[INGESTION] Erreurs lors de l'import: {batch.number_errors}")
                print(batch.failed_objects)

        print("[INGESTION] Import terminé.")


def main():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_dir = project_root / "data" / "knowledge"

    print(f"[INGESTION] Lecture des fiches dans {knowledge_dir}")
    fiches = load_markdown_files(knowledge_dir)
    print(f"[INGESTION] Fichiers markdown détectés: {len(fiches)}")

    chunks = build_chunk_objects(fiches)
    print(f"[INGESTION] Chunks générés: {len(chunks)}")

    # Optionnel: aperçu des premiers chunks
    print("[INGESTION] Exemple de chunk:")
    if chunks:
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

    ingest_chunks_into_weaviate(chunks)


if __name__ == "__main__":
    main()
