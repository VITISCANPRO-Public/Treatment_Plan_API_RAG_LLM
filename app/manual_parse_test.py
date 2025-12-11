from app.rag_pipeline import parse_llm_structured_response

if __name__ == "__main__":
    # Exemple de réponse un peu sale comme un vrai LLM
    fake_output = """
    ```json
    {
      "diagnostic": "Le mildiou est détecté sur la parcelle, avec une pression modérée.",
      "treatment_actions": [
        "Appliquer un traitement à base de cuivre selon la dose recommandée.",
        "Surveiller l'apparition de nouvelles taches après le traitement."
      ],
      "preventive_actions": [
        "Améliorer l'aération de la vigne.",
        "Éviter les arrosages excessifs."
      ],
      "warnings": [
        "Respecter les délais avant récolte indiqués sur l'étiquette du produit."
      ]
    }
    ```
    """

    parsed = parse_llm_structured_response(fake_output)
    print("=== PARSED ===")
    for k, v in parsed.items():
        print(f"{k}: {v}")
