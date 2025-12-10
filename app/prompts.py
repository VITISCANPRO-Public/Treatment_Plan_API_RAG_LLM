from typing import List, Dict


def build_treatment_prompt(
    cnn_label: str,
    disease_name_fr: str,
    mode: str,
    severity: str,
    area_m2: float,
    season: str,
    context_chunks: List[Dict[str, str]],
) -> str:
    """
    Construit le prompt envoyé au LLM à partir des chunks de contexte.
    context_chunks est une liste de dicts avec au minimum la clé 'text'.
    """

    docs_text = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)

    prompt = f"""
Vous êtes un conseiller viticole en France.

Vous devez proposer un plan d'action court, synthétique et scientifique pour un viticulteur.

Vous devez répondre uniquement à partir des informations fournies dans les extraits de documentation.
Si les informations sont insuffisantes, dites-le explicitement et conseillez de consulter un technicien local.

Contexte:
- Maladie détectée : {disease_name_fr} (label CNN : {cnn_label})
- Mode de conduite : {mode}
- Gravité observée : {severity}
- Surface concernée : {area_m2} m²
- Période de l'année : {season}

Extraits de documentation (à utiliser obligatoirement) :
{docs_text}

Consignes de réponse :
- Répondez en français.
- Adressez-vous au viticulteur en utilisant "vous".
- Soyez factuel, prudent et scientifique.
- Structurez la réponse en 4 parties :
  1. Diagnostic rapide
  2. Actions de traitement immédiates
  3. Mesures préventives
  4. Rappels de sécurité et de réglementation
"""

    return prompt.strip()
