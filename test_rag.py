from app.weaviate_client import weaviate_client
from app.rag_pipeline import search_treatment_chunks

def run():
    tests = [
        ("downy_mildew", "conventionnel", "moderee"),
        ("Grape_Downy_mildew_leaf", "conventionnel", "moderee"),
    ]

    with weaviate_client() as client:
        for disease_input, mode, severity in tests:
            chunks = search_treatment_chunks(client, disease_input, mode, severity, top_k=5)
            print("\n--- TEST ---")
            print("disease_input:", disease_input, "| mode:", mode, "| severity:", severity)
            print("chunks:", len(chunks))
            if chunks:
                c0 = chunks[0]
                print("ex0.section:", c0.get("section"))
                print("ex0.nom_fr:", c0.get("nom_fr"))
                print("ex0.distance:", c0.get("distance"))
                print("ex0.text[:200]:", (c0.get("text") or "")[:200])

if __name__ == "__main__":
    run()
