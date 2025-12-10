from datetime import datetime
from typing import Dict, Any

from .dosage_rules import compute_dosage
from .weaviate_client import weaviate_client, search_treatment_chunks
from .prompts import build_treatment_prompt


def infer_season_from_date(date_iso: str) -> str:
    """
    Déduit une saison simplifiée à partir d'une date ISO (YYYY-MM-DD).
    C'est approximatif mais suffisant pour le contexte.
    """
    if not date_iso:
        return "inconnue"

    try:
        month = datetime.fromisoformat(date_iso).month
    except ValueError:
        return "inconnue"

    if month in (12, 1, 2):
        return "hiver"
    if month in (3, 4, 5):
        return "printemps"
    if month in (6, 7, 8):
        return "été"
    if month in (9, 10, 11):
        return "automne"
    return "inconnue"


def fake_llm_call(prompt: str) -> str:
    """
    Pour l'instant, on simule l'appel au LLM.
    On remplacera cette fonction par un vrai appel à l'API plus tard.
    """
    return (
        "Diagnostic rapide : le contexte fourni indique une maladie foliaire probable.\n\n"
        "Actions de traitement immédiates : appliquez un traitement adapté en respectant les doses recommandées.\n\n"
        "Mesures préventives : surveillez régulièrement vos parcelles et favorisez l'aération de la végétation.\n\n"
        "Rappels de sécurité : portez vos équipements de protection individuelle et respectez la réglementation en vigueur."
    )


def generate_treatment_advice(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline principal :
    1) Déduit la saison à partir de la date.
    2) Va chercher les chunks de connaissance pertinents dans Weaviate.
    3) Construit un prompt RAG et (bientôt) appelle un LLM.
    4) Calcule les dosages via compute_dosage.
    5) Retourne une réponse structurée pour l’API.
    """

    cnn_label = payload["cnn_label"]
    mode = payload["mode"]
    severity = payload["severity"]
    area_m2 = float(payload["area_m2"])
    date_iso = payload.get("date_iso", "")

    # 1. Saison
    season = infer_season_from_date(date_iso)

    # 2. Récupération des chunks dans Weaviate.
    #    On utilise le context manager pour que la connexion soit
    #    proprement fermée (pas de warning "connection not closed").
    with weaviate_client() as client:
        chunks = search_treatment_chunks(client, cnn_label, mode, severity)

    # 3. Si aucun chunk, on renvoie un plan basé uniquement sur les règles de dosage.
    if not chunks:
        dosage = compute_dosage(cnn_label, mode, area_m2)
        return {
            "cnn_label": cnn_label,
            "mode": mode,
            "area_m2": area_m2,
            "severity": severity,
            "season": season,
            "treatment_plan": dosage,
            "advice_text": (
                "Les informations disponibles sur cette maladie sont insuffisantes "
                "pour proposer un traitement fiable à partir de la base de connaissances. "
                "Nous vous recommandons de consulter un technicien viticole local."
            ),
            "preventive_actions": [],
            "warnings": [
                "Ces recommandations sont indicatives.",
                "Vérifiez la réglementation locale et les notices des produits avant application.",
            ],
        }

    # 4. Construction du prompt à partir des chunks RAG.
    disease_name_fr = chunks[0].get("nom_fr", cnn_label)

    prompt = build_treatment_prompt(
        cnn_label=cnn_label,
        disease_name_fr=disease_name_fr,
        mode=mode,
        severity=severity,
        area_m2=area_m2,
        season=season,
        context_chunks=[{"text": c["text"]} for c in chunks],
    )

    # 5. (Pour l’instant) on appelle un faux LLM.
    advice_text = fake_llm_call(prompt)

    # 6. Calcul des dosages avec les règles métiers.
    dosage = compute_dosage(cnn_label, mode, area_m2)

    # 7. Réponse structurée pour l’API.
    return {
        "cnn_label": cnn_label,
        "mode": mode,
        "area_m2": area_m2,
        "severity": severity,
        "season": season,
        "treatment_plan": dosage,
        "advice_text": advice_text,
        "preventive_actions": [],
        "warnings": [
            "Ces recommandations sont indicatives.",
            "Vérifiez la réglementation locale et les notices des produits avant application.",
        ],
    }
