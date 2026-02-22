"""
ingestion.py — Loads markdown knowledge files and indexes them into Weaviate.
Run this script once before starting the API to populate the knowledge base.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import frontmatter
import weaviate
import weaviate.classes as wvc

from app.weaviate_client import weaviate_client, get_embedder

COLLECTION_NAME = "VitiScanKnowledge"


def load_markdown_files(knowledge_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads all .md files from the data/knowledge directory.

    Returns:
        List of dicts with keys: path, meta, content
    """
    md_files = sorted(knowledge_dir.glob("*.md"))
    fiches: List[Dict[str, Any]] = []

    for md_path in md_files:
        post = frontmatter.load(md_path)
        fiches.append({
            "path":    str(md_path),
            "meta":    dict(post.metadata),
            "content": post.content,
        })

    return fiches


def split_markdown_sections(content: str) -> List[Dict[str, str]]:
    """
    Splits markdown content into sections based on level-1 headings '# '.

    Returns:
        List of dicts with keys: section_title, text
    """
    lines = content.splitlines()
    sections: List[Dict[str, str]] = []

    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        heading_match = re.match(r"^#\s+(.*)", line.strip())
        if heading_match:
            if current_title is not None and current_lines:
                sections.append({
                    "section_title": current_title,
                    "text": "\n".join(current_lines).strip(),
                })
            current_title = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_title is not None and current_lines:
        sections.append({
            "section_title": current_title,
            "text": "\n".join(current_lines).strip(),
        })

    return sections


def ensure_collection(client: weaviate.WeaviateClient):
    """
    Creates the VitiScanKnowledge collection if it does not already exist.
    Vectors are self-provided (embeddings computed locally).
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
                    name="disease_name",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="type",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="category",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="farming_mode",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
        )
        return coll


def build_chunk_objects(fiches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms markdown fiches into a list of chunks ready for indexing.

    Each chunk contains:
    - text: full chunk text (section title + content)
    - section: section name
    - disease_id, cnn_label, disease_name, type, category, farming_mode: from frontmatter
    """
    all_chunks: List[Dict[str, Any]] = []

    for fiche in fiches:
        meta    = fiche["meta"]
        content = fiche["content"]
        sections = split_markdown_sections(content)

        disease_id   = meta.get("id")
        cnn_label    = meta.get("cnn_label")
        disease_name = meta.get("disease_name")
        disease_type = meta.get("type")
        category     = meta.get("category")
        farming_mode = meta.get("farming_mode") or []

        # Store farming_mode as a simple string for easier filtering
        farming_mode_str = ", ".join(farming_mode)

        for section in sections:
            full_text = f"{section['section_title']}\n\n{section['text']}".strip()

            all_chunks.append({
                "text":         full_text,
                "section":      section["section_title"],
                "disease_id":   disease_id,
                "cnn_label":    cnn_label,
                "disease_name": disease_name,
                "type":         disease_type,
                "category":     category,
                "farming_mode": farming_mode_str,
            })

    return all_chunks


def ingest_chunks_into_weaviate(chunks: List[Dict[str, Any]]):
    """
    Sends all chunks into Weaviate with SentenceTransformer embeddings.
    Uses the weaviate_client() context manager to open/close the connection.
    """
    with weaviate_client() as client:
        collection = ensure_collection(client)
        embedder   = get_embedder()

        print(f"[INGESTION] Indexing {len(chunks)} chunks...")

        with collection.batch.dynamic() as batch:
            for idx, chunk in enumerate(chunks, start=1):
                vector = embedder.encode(chunk["text"]).tolist()

                batch.add_object(
                    properties={
                        "text":         chunk["text"],
                        "section":      chunk["section"],
                        "disease_id":   chunk["disease_id"],
                        "cnn_label":    chunk["cnn_label"],
                        "disease_name": chunk["disease_name"],
                        "type":         chunk["type"],
                        "category":     chunk["category"],
                        "farming_mode": chunk["farming_mode"],
                    },
                    vector=vector,
                )

                if idx % 20 == 0:
                    print(f"[INGESTION] {idx} chunks sent...")

            if batch.number_errors > 0:
                print(f"[INGESTION] Errors during import: {batch.number_errors}")
                print(batch.failed_objects)

        print("[INGESTION] Import complete.")


def main():
    project_root  = Path(__file__).resolve().parents[1]
    knowledge_dir = project_root / "data" / "knowledge"

    print(f"[INGESTION] Reading knowledge files from {knowledge_dir}")
    fiches = load_markdown_files(knowledge_dir)
    print(f"[INGESTION] Markdown files found: {len(fiches)}")

    chunks = build_chunk_objects(fiches)
    print(f"[INGESTION] Chunks generated: {len(chunks)}")

    print("[INGESTION] Sample chunk:")
    if chunks:
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

    ingest_chunks_into_weaviate(chunks)


if __name__ == "__main__":
    main()