import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import re
import os
from app.dosage_rules import compute_dosage
from app.weaviate_client_local import weaviate_client, search_treatment_chunks
from app.prompts_local import build_treatment_prompt
from app.llm_client import call_llm, LLMError

disable_weaviate = os.getenv("DISABLE_WEAVIATE", "0") == "1"

print(">>> RAG PIPELINE LOCAL CHARGÉ <<<")

def _to_str_list(value: Any) -> List[str]:
    """
    Convertit une valeur en liste de chaînes propres.
    - list -> nettoie
    - str  -> split si multi-lignes / puces, sinon [str]
    - autre -> []
    """
    if value is None:
        return []

    if isinstance(value, list):
        out = []
        for v in value:
            s = str(v).strip()
            if s:
                out.append(s)
        return out

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Si le LLM a renvoyé une liste sous forme de texte (puces ou lignes)
        if "\n" in s or s.lstrip().startswith(("-", "•")):
            lines = []
            for line in s.splitlines():
                line = line.strip()
                line = line.lstrip("-• ").strip()
                if line:
                    lines.append(line)
            return lines
        return [s]

    return []

def parse_llm_structured_response(raw: str) -> Dict[str, Any]:
    """
    Parsing robuste :
    - retire ```json
    - extrait le premier objet JSON {...}
    - tente json.loads (avec support JSON doublement encodé)
    - fallback heuristique si JSON invalide
    - normalise les champs en listes
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

    # Retire fences éventuelles
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        text = text.replace("json\n", "").replace("json\r\n", "").strip()

    # Extraction JSON
    candidate = _extract_first_json_object(text)
    if not candidate:
        # JSON tronqué : on tente au moins d'extraire des champs avec la méthode heuristique
        data = _heuristic_parse_from_text(text)
        if any([data.get("diagnostic"), data.get("treatment_actions"), data.get("preventive_actions"), data.get("warnings")]):
            return {
                "diagnostic": (data.get("diagnostic") or "").strip(),
                "treatment_actions": _to_str_list(data.get("treatment_actions")),
                "preventive_actions": _to_str_list(data.get("preventive_actions")),
                "warnings": _to_str_list(data.get("warnings")),
            }
        return default

    # Nettoyages légers
    candidate = (
        candidate.replace("```json", "")
        .replace("```", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .strip()
    )
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # retire virgules finales

    # Parsing principal + double encodage
    try:
        data = json.loads(candidate)
        if isinstance(data, str):
            data = json.loads(data)
    except Exception:
        # fallback heuristique si JSON invalide (ex: tronqué)
        data = _heuristic_parse_from_text(text)

    if not isinstance(data, dict):
        return default

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

CNN_LABEL_ALIASES = {
    # FR (UI)
    "mildiou": "Grape_Downy_mildew_leaf",
    "oïdium": "Grape_Powdery_mildew_leaf",
    "oidium": "Grape_Powdery_mildew_leaf",
    "tache brune": "Grape_Brown_spot_leaf",
    "tache_brune": "Grape_Brown_spot_leaf",
    "anthracnose": "Grape_Anthracnose_leaf",
    "acariens": "Grape_Mites_leaf_disease",
    "mites": "Grape_Mites_leaf_disease",
    "shot_hole": "Grape_shot_hole_leaf_disease",
    "sain": "Grape_Normal_leaf",
    "normal": "Grape_Normal_leaf",
    "downy_mildew": "Grape_Downy_mildew_leaf",
    "powdery_mildew": "Grape_Powdery_mildew_leaf",
    "brown_spot": "Grape_Brown_spot_leaf",
}

def normalize_cnn_label(raw: str) -> str:
    if not raw:
        return raw
    key = raw.strip()
    # si déjà un label CNN exact, on le garde
    if key.startswith("Grape_"):
        return key
    return CNN_LABEL_ALIASES.get(key.lower(), key)

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Essaie d'extraire un objet JSON {...} depuis un texte (même s'il y a du bruit autour).
    Retourne une string JSON candidate, ou None.
    """
    if not text:
        return None

    # Enlève les fences si présents
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Cherche le premier '{' et le dernier '}' après celui-ci
    start = cleaned.find("{")
    if start == -1:
        return None

    end = cleaned.rfind("}")
    if end == -1 or end <= start:
        return None

    return cleaned[start:end + 1]


def _safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return []


def _heuristic_parse_from_text(text: str) -> Dict[str, Any]:
    """
    Fallback : si le JSON est invalide, on tente d'extraire des champs avec regex.
    """
    out: Dict[str, Any] = {
        "diagnostic": "",
        "treatment_actions": [],
        "preventive_actions": [],
        "warnings": [],
    }

    if not text:
        return out

    # Diagnostic
    m = re.search(r'"diagnostic"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    if m:
        out["diagnostic"] = m.group(1).replace("\\n", "\n").strip()

    # Listes
    def extract_list(key: str) -> List[str]:
        m2 = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        if not m2:
            return []
        inside = m2.group(1)
        return [s.replace("\\n", "\n").strip() for s in re.findall(r'"([^"]+)"', inside)]

    out["treatment_actions"] = extract_list("treatment_actions")
    out["preventive_actions"] = extract_list("preventive_actions")
    out["warnings"] = extract_list("warnings")
    return out

def generate_treatment_advice(payload: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
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

    cnn_label = normalize_cnn_label(payload["cnn_label"])

    # 2. Récupération des chunks dans Weaviate
    chunks = []
    if disable_weaviate:
        chunks = [
            {
                "text": "Mildiou (downy mildew) : privilégier une intervention rapide après pluie, surveiller les foyers, renouveler la protection selon conditions. Respecter les doses homologuées et la réglementation locale.",
                "section": "Traitement",
                "disease_id": "Grape_Downy_mildew_leaf",
                "nom_fr": "Mildiou"
            },
            {
                "text": "Prévention : aérer le feuillage (effeuillage si nécessaire), limiter l’humidité, surveiller météo et pression maladie, éviter excès d’azote.",
                "section": "Prévention",
                "disease_id": "Grape_Downy_mildew_leaf",
                "nom_fr": "Mildiou"
            }
        ]
    else:
        with weaviate_client() as client:
            chunks = search_treatment_chunks(
                client=client,
                disease_input=cnn_label,
                mode=mode,
                severity=severity,
                top_k=8,
            )

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

    if debug:
        print("\n===== PROMPT ENVOYÉ AU LLM =====\n")
        print(prompt)

    # 5. Appel au LLM Hugging Face via le wrapper + parsing structuré.
    try:
        raw_llm_text = call_llm(
            prompt,
            max_new_tokens=700,
            temperature=0.2,
            top_p=0.9
        )

        if debug:
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
        if debug:
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

    # Warnings de base + warnings issus du LLM
    base_warnings = [
        "Ces recommandations sont indicatives.",
        "Vérifiez la réglementation locale et les notices des produits avant application.",
    ]

    treatment_actions = parsed.get("treatment_actions") or []
    preventive_actions = parsed.get("preventive_actions") or []
    llm_warnings = parsed.get("warnings") or []
    warnings = base_warnings + llm_warnings
    diagnostic_text = parsed.get("diagnostic") or ""

    result = {
        "cnn_label": cnn_label,
        "mode": mode,
        "area_m2": area_m2,
        "severity": severity,
        "season": season,
        "diagnostic": diagnostic_text,
        "treatment_actions": treatment_actions,
        "preventive_actions": preventive_actions,
        "warnings": warnings,
    }

    if debug:
        result["raw_llm_output"] = raw_llm_text    
        print("\n===== RÉPONSE FINALE RETOURNÉE PAR generate_treatment_advice =====\n")
        print(result)

    return result