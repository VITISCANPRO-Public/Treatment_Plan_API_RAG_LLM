from datetime import datetime
from typing import Dict, Any

from .dosage_rules import compute_dosage
from .weaviate_client import get_weaviate_client, search_treatment_chunks
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
    On remplacera cette fonction par un vrai appel à l'API HF plus tard.
    """
    return (
        "Diagnostic rapide : le contexte fourni indique une maladie foliaire probable.\n\n"
        "Actions de traitement immédiates : appliquez un traitement adapté en respectant les doses recommandées.\n\n"
        "Mesures préventives : surveillez régulièrement vos parcelles et favorisez l'aération de la végétation.\n\n"
        "Rappels de sécurité : portez vos équipements de protection individuelle et respectez la réglementation en vigueur."
    )


def generate_treatment_advice(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fonction principale du pipeline RAG.

    Entrée (payload) : dict contenant au minimum :
    - cnn_label : str
    - mode : "bio" ou "conventionnel"
    - severity : "faible" | "moderee" | "forte"
    - area_m2 : float
    - date_iso : str (optionnel)

    Sortie : dict prêt à être renvoyé par l'API FastAPI.
    """

    cnn_label = payload["cnn_label"]
    mode = payload["mode"]
    severity = payload["severity"]
    area_m2 = float(payload["area_m2"])
    date_iso = payload.get("date_iso") or ""

    season = infer_season_from_date(date_iso)

    # 1. Récupération des chunks dans Weaviate (pour l'instant renvoie une liste vide).
    client = get_weaviate_client()
    chunks = search_treatment_chunks(client, cnn_label, mode, severity)

    # 2. Si aucun chunk, on renvoie un message de prudence + dosage éventuel.
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

    # 3. Si on a des chunks, on construit un prompt et on appelle (fictivement) le LLM.
    prompt = build_treatment_prompt(
        cnn_label=cnn_label,
        disease_name_fr=cnn_label,  # on utilisera un vrai nom FR plus tard
        mode=mode,
        severity=severity,
        area_m2=area_m2,
        season=season,
        context_chunks=chunks,
    )

    advice_text = fake_llm_call(prompt)

    # 4. Calcul des dosages.
    dosage = compute_dosage(cnn_label, mode, area_m2)

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
