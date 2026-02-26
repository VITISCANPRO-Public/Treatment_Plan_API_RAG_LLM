"""
test_rag.py — Manual validation script for the RAG pipeline.

This script is NOT an automated pytest test.
It requires a running Weaviate instance.

To be run manually in two situations:
  1. After the initial ingestion   → check that the chunks are correctly indexed
  2. After a Weaviate migration    → check that vector search is working

Usage:
    python scripts/test_rag.py
"""

from app.weaviate_client import weaviate_client, search_treatment_chunks


def run():
    tests = [
        ("plasmopara_viticola",         "conventional", "moderate"),
        ("erysiphe_necator",            "organic",      "high"),
        ("elsinoe_ampelina",            "conventional", "low"),
        ("guignardia_bidwellii",        "conventional", "moderate"),
        ("phaeomoniella_chlamydospora", "organic",      "high"),
        ("colomerus_vitis",             "conventional", "low"),
        ("healthy",                     "conventional", "low"),
    ]

    with weaviate_client() as client:
        for disease_input, mode, severity in tests:
            chunks = search_treatment_chunks(
                client=client,
                disease_input=disease_input,
                mode=mode,
                severity=severity,
                top_k=5,
            )
            print("\n--- TEST ---")
            print(f"disease_input : {disease_input}")
            print(f"mode          : {mode}")
            print(f"severity      : {severity}")
            print(f"chunks found  : {len(chunks)}")

            if chunks:
                c0 = chunks[0]
                print(f"first chunk section      : {c0.get('section')}")
                print(f"first chunk disease_name : {c0.get('disease_name')}")
                print(f"first chunk distance     : {c0.get('distance')}")
                print(f"first chunk text[:200]   : {(c0.get('text') or '')[:200]}")
            else:
                print("No chunks retrieved — check ingestion and Weaviate connection.")


if __name__ == "__main__":
    run()