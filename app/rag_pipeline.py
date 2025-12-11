import json
from datetime import datetime
from typing import Dict, Any, List

from app.dosage_rules import compute_dosage
from app.weaviate_client import weaviate_client, search_treatment_chunks
from app.prompts import build_treatment_prompt
from app.llm_client import call_llm, LLMError

DEBUG = False

def _to_str_list(value: Any) -> List[str]:
    """
    Convertit une valeur en liste de chaînes propres.
    - Si c'est déjà une liste -> nettoie chaque élément.
    - Si c'est une chaîne -> renvoie [chaine].
    - Sinon -> liste vide.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def parse_llm_structured_response(raw: str) -> Dict[str, Any]:
    """
    Parse la réponse du LLM censée être un JSON avec :
    - diagnostic: str
    - treatment_actions: List[str]
    - preventive_actions: List[str]
    - warnings: List[str]

    Si le JSON est invalide ou incomplet, on renvoie une structure par défaut
    en mettant le texte brut dans "diagnostic".
    """
    default = {
        "diagnostic": raw.strip() if raw else "",
        "treatment_actions": [],
        "preventive_actions": [],
        "warnings": [],
    }

    if not raw or not raw.strip():
        return default

    text = raw.strip()

    # 1) On enlève d'éventuels backticks ou balises ```json
    # Cas fréquent : le modèle entoure la réponse avec ```json ... ```
    if "```" in text:
        # On garde ce qu'il y a entre le premier et le dernier ```
        parts = text.split("```")
        # souvent : ["", "json\n{...}", ""]
        if len(parts) >= 3:
            text = parts[1]
        # On vire un éventuel préfixe "json\n"
        text = text.replace("json\n", "").replace("json\r\n", "").strip()

    # 2) On essaie de parser le JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Dernier essai : parfois le JSON est au milieu d'un texte
        # On essaye de récupérer la première accolade jusqu'à la dernière.
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except Exception:
            # On ne casse pas la pipeline : on garde le texte brut
            return default

    # 3) Normalisation des champs
    diagnostic = str(data.get("diagnostic", "")).strip() or default["diagnostic"]
    treatment_actions = _to_str_list(data.get("treatment_actions"))
    preventive_actions = _to_str_list(data.get("preventive_actions"))
    warnings = _to_str_list(data.get("warnings"))

    return {
        "diagnostic": diagnostic,
        "treatment_actions": treatment_actions,
        "preventive_actions": preventive_actions,
        "warnings": warnings,
    }

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

def generate_treatment_advice(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline principal :
    1) Déduit la saison à partir de la date.
    2) Va chercher les chunks de connaissance pertinents dans Weaviate.
    3) Construit un prompt RAG et appelle un LLM.
    4) Calcule les dosages via compute_dosage.
    5) Retourne une réponse structurée pour l'API.
    """
    cnn_label = payload["cnn_label"]
    mode = payload["mode"]
    severity = payload["severity"]
    area_m2 = float(payload["area_m2"])
    date_iso = payload.get("date_iso", "")

    season = infer_season_from_date(date_iso)

    # 2. Récupération des chunks dans Weaviate
    with weaviate_client() as client:
        chunks = search_treatment_chunks(client, cnn_label, mode, severity)

    # 3. Si aucun chunk, on renvoie un plan basé uniquement sur les règles de dosage.
    dosage = compute_dosage(cnn_label, mode, area_m2)
    if not chunks:
        return {
            "cnn_label": cnn_label,
            "mode": mode,
            "area_m2": area_m2,
            "severity": severity,
            "season": season,
            "treatment_plan": dosage,
            "diagnostic": (
                "Les informations disponibles sur cette maladie sont insuffisantes "
                "pour proposer un traitement fiable à partir de la base de connaissances. "
                "Nous vous recommandons de consulter un technicien viticole local."
            ),
            "treatment_actions": [],
            "preventive_actions": [],
            "warnings": [
                "Ces recommandations sont indicatives.",
                "Vérifiez la réglementation locale et les notices des produits avant application.",
            ],
            "advice_text": "",
            "raw_llm_output": "",
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

    if DEBUG:
        print("\n===== PROMPT ENVOYÉ AU LLM =====\n")
        print(prompt)

    # 5. Appel au LLM Hugging Face via le wrapper + parsing structuré.
    try:
        raw_llm_text = call_llm(prompt)

        if DEBUG:
            print("\n===== RAW LLM TEXT =====\n")
            print(raw_llm_text)

        parsed = parse_llm_structured_response(raw_llm_text)

        # Fallback de sécurité si le parsing a échoué partiellement
        if not parsed.get("diagnostic"):
            fallback_text = (
                "Diagnostic rapide : la situation nécessite un avis technique.\n"
                "Je n'ai pas pu générer une recommandation détaillée automatiquement.\n"
                "Veuillez consulter un conseiller viticole local.\n"
            )
            parsed["diagnostic"] = fallback_text

    except LLMError as e:
        if DEBUG:
            print("\n===== ERREUR LLM =====\n")
            print(f"Erreur LLM : {e}")

        # Fallback de sécurité : si le LLM plante, on revient à un message simple
        fallback_text = (
            "Diagnostic rapide : la situation nécessite un avis technique.\n"
            "Je n'ai pas pu générer une recommandation détaillée automatiquement.\n"
            "Veuillez consulter un conseiller viticole local.\n"
            f"(Détail technique : {e})\n"
        )
        parsed = {
            "diagnostic": fallback_text,
            "treatment_actions": [],
            "preventive_actions": [],
            "warnings": [],
        }
        raw_llm_text = fallback_text

    # 6. Calcul des dosages avec les règles métiers.
    dosage = compute_dosage(cnn_label, mode, area_m2)

    # 7. Construction d'un texte synthétique (utile côté UI) à partir des champs structurés.
    advice_text = (
        f"Diagnostic : {parsed['diagnostic']}\n\n"
        "Actions de traitement :\n"
        + "\n".join(f"- {a}" for a in parsed["treatment_actions"]) + "\n\n"
        "Mesures préventives :\n"
        + "\n".join(f"- {a}" for a in parsed["preventive_actions"]) + "\n"
    )

    # Warnings de base + warnings issus du LLM
    base_warnings = [
        "Ces recommandations sont indicatives.",
        "Vérifiez la réglementation locale et les notices des produits avant application.",
    ]
    all_warnings = base_warnings + parsed["warnings"]

    result = {
        "cnn_label": cnn_label,
        "mode": mode,
        "area_m2": area_m2,
        "severity": severity,
        "season": season,
        "treatment_plan": dosage,
        "diagnostic": parsed["diagnostic"],
        "treatment_actions": parsed["treatment_actions"],
        "preventive_actions": parsed["preventive_actions"],
        "warnings": all_warnings,
        "advice_text": advice_text,
        "raw_llm_output": raw_llm_text,
    }

    if DEBUG:
        print("\n===== RÉPONSE FINALE RETOURNÉE PAR generate_treatment_advice =====\n")
        print(result)

    return result