"""
rag_pipeline.py — Main RAG pipeline for treatment plan generation.

Pipeline steps:
1. Infer season from date
2. Retrieve relevant knowledge chunks from Weaviate
3. Build RAG prompt and call LLM
4. Compute dosage via dosage_rules
5. Return structured response for the API
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.dosage_rules import compute_dosage
from app.llm_client import LLMError, call_llm
from app.prompts import build_treatment_prompt
from app.weaviate_client import search_treatment_chunks, weaviate_client

DEBUG = False

# ── Disease label mapping (INRAE labels → English names) ──────────────────────

DISEASE_NAMES: Dict[str, str] = {
    "colomerus_vitis":             "Erinose",
    "elsinoe_ampelina":            "Anthracnose",
    "erysiphe_necator":            "Powdery Mildew",
    "guignardia_bidwellii":        "Black Rot",
    "healthy":                     "Healthy",
    "phaeomoniella_chlamydospora": "Esca",
    "plasmopara_viticola":         "Downy Mildew",
}


# ── Helper functions ───────────────────────────────────────────────────────────

def infer_season_from_date(date_iso: str) -> str:
    """
    Infers a simplified season from an ISO date string (YYYY-MM-DD).

    Returns:
        Season string: 'winter', 'spring', 'summer', 'autumn', or 'unknown'
    """
    if not date_iso:
        return "unknown"

    try:
        month = datetime.fromisoformat(date_iso).month
    except ValueError:
        return "unknown"

    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "unknown"


def _to_str_list(value: Any) -> List[str]:
    """
    Converts a value to a clean list of strings.
    - list  → cleans each item
    - str   → splits on newlines/bullets if needed, else wraps in list
    - other → returns []
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "\n" in s or s.lstrip().startswith(("-", "•")):
            return [
                line.strip().lstrip("-• ").strip()
                for line in s.splitlines()
                if line.strip().lstrip("-• ").strip()
            ]
        return [s]

    return []


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extracts the first JSON object {...} from a text string.

    Returns:
        JSON candidate string, or None if not found
    """
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return None

    end = cleaned.rfind("}")
    if end == -1 or end <= start:
        return None

    return cleaned[start:end + 1]


def _heuristic_parse_from_text(text: str) -> Dict[str, Any]:
    """
    Fallback parser: extracts fields from text using regex when JSON is invalid.
    """
    out: Dict[str, Any] = {
        "diagnostic":        "",
        "treatment_actions":  [],
        "preventive_actions": [],
        "warnings":           [],
    }

    if not text:
        return out

    m = re.search(r'"diagnostic"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    if m:
        out["diagnostic"] = m.group(1).replace("\\n", "\n").strip()

    def extract_list(key: str) -> List[str]:
        m2 = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        if not m2:
            return []
        return [
            s.replace("\\n", "\n").strip()
            for s in re.findall(r'"([^"]+)"', m2.group(1))
        ]

    out["treatment_actions"]  = extract_list("treatment_actions")
    out["preventive_actions"] = extract_list("preventive_actions")
    out["warnings"]           = extract_list("warnings")
    return out


def parse_llm_structured_response(raw: str) -> Dict[str, Any]:
    """
    Robust LLM response parser:
    - Removes ```json fences
    - Extracts first JSON object {...}
    - Attempts json.loads (with double-encoding support)
    - Falls back to heuristic regex parsing if JSON is invalid
    - Normalizes all fields to string lists

    Args:
        raw: Raw LLM output string

    Returns:
        Dict with keys: diagnostic, treatment_actions, preventive_actions, warnings
    """
    default = {
        "diagnostic":        raw.strip() if raw else "",
        "treatment_actions":  [],
        "preventive_actions": [],
        "warnings":           [],
    }

    if not raw or not raw.strip():
        return default

    text = raw.strip()

    # Remove code fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        text = text.replace("json\n", "").replace("json\r\n", "").strip()

    # Extract JSON candidate
    candidate = _extract_first_json_object(text)
    if not candidate:
        data = _heuristic_parse_from_text(text)
        if any([data.get("diagnostic"), data.get("treatment_actions"),
                data.get("preventive_actions"), data.get("warnings")]):
            return {
                "diagnostic":        (data.get("diagnostic") or "").strip(),
                "treatment_actions":  _to_str_list(data.get("treatment_actions")),
                "preventive_actions": _to_str_list(data.get("preventive_actions")),
                "warnings":           _to_str_list(data.get("warnings")),
            }
        return default

    # Light cleanup
    candidate = (
        candidate
        .replace("```json", "")
        .replace("```", "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2019", "'")
        .strip()
    )
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    # Parse JSON (with double-encoding fallback)
    try:
        data = json.loads(candidate)
        if isinstance(data, str):
            data = json.loads(data)
    except Exception:
        data = _heuristic_parse_from_text(text)

    if not isinstance(data, dict):
        return default

    return {
        "diagnostic":        str(data.get("diagnostic", "")).strip() or default["diagnostic"],
        "treatment_actions":  _to_str_list(data.get("treatment_actions")),
        "preventive_actions": _to_str_list(data.get("preventive_actions")),
        "warnings":           _to_str_list(data.get("warnings")),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def generate_treatment_advice(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RAG pipeline:
    1. Infer season from date
    2. Retrieve relevant knowledge chunks from Weaviate
    3. Build RAG prompt and call LLM
    4. Compute dosage via dosage_rules
    5. Return structured response for the API

    Args:
        payload: Dict with keys: cnn_label, mode, severity, area_m2, date_iso

    Returns:
        Structured treatment plan dict
    """
    cnn_label = payload["cnn_label"]
    mode      = str(payload["mode"]).strip().lower()
    severity  = str(payload["severity"]).strip().lower()
    area_m2   = float(payload["area_m2"])
    date_iso  = payload.get("date_iso", "")

    season       = infer_season_from_date(date_iso)
    disease_name = DISEASE_NAMES.get(cnn_label, cnn_label)

    # ── Step 1: Retrieve chunks from Weaviate ──────────────────────────────────
    with weaviate_client() as client:
        chunks = search_treatment_chunks(
            client=client,
            disease_input=cnn_label,
            mode=mode,
            severity=severity,
            top_k=8,
        )
        # Fallback: retry without mode filter
        if not chunks:
            chunks = search_treatment_chunks(
                client=client,
                disease_input=cnn_label,
                mode=None,
                severity=severity,
                top_k=8,
            )

    if DEBUG:
        print(f"\n[RAG] {len(chunks)} chunks retrieved for '{cnn_label}'")

    # ── Step 2: Compute dosage ─────────────────────────────────────────────────
    dosage = compute_dosage(cnn_label, mode, area_m2, severity=severity)
    if not dosage:
        dosage = {"note": "No dosage rule available for this disease/mode/severity combination."}

    # ── Step 3: Fallback chunks if Weaviate returned nothing ───────────────────
    if not chunks:
        chunks = [{
            "text": (
                "No relevant extract found in the knowledge base. "
                "Base recommendations on dosage rules and general best practices only."
            )
        }]

    # ── Step 4: Build prompt and call LLM ─────────────────────────────────────
    prompt = build_treatment_prompt(
        cnn_label=cnn_label,
        disease_name=disease_name,
        mode=mode,
        severity=severity,
        area_m2=area_m2,
        season=season,
        context_chunks=[{"text": c["text"]} for c in chunks],
    )

    if DEBUG:
        print("\n===== PROMPT SENT TO LLM =====\n")
        print(prompt)

    # ── Step 5: Parse LLM response ─────────────────────────────────────────────
    try:
        raw_llm_text = call_llm(prompt, max_new_tokens=700, temperature=0.2, top_p=0.9)

        if DEBUG:
            print("\n===== RAW LLM OUTPUT =====\n")
            print(raw_llm_text)

        parsed = parse_llm_structured_response(raw_llm_text)

        if not parsed.get("diagnostic"):
            parsed["diagnostic"] = (
                "The situation requires technical assessment. "
                "No detailed recommendation could be generated automatically. "
                "Please consult a local viticulture advisor."
            )

    except LLMError as e:
        if DEBUG:
            print(f"\n===== LLM ERROR =====\n{e}")

        fallback_text = (
            "The situation requires technical assessment. "
            "No detailed recommendation could be generated automatically. "
            "Please consult a local viticulture advisor. "
            f"(Technical detail: {e})"
        )
        parsed = {
            "diagnostic":        fallback_text,
            "treatment_actions":  [],
            "preventive_actions": [],
            "warnings":           [],
        }
        raw_llm_text = fallback_text

    # ── Step 6: Build final result ─────────────────────────────────────────────
    base_warnings = [
        "These recommendations are indicative only.",
        "Always verify local regulations and product labels before application.",
    ]

    result = {
        "cnn_label":          cnn_label,
        "disease_name":       disease_name,
        "mode":               mode,
        "area_m2":            area_m2,
        "severity":           severity,
        "season":             season,
        "treatment_plan":     dosage,
        "diagnostic":         parsed.get("diagnostic") or "",
        "treatment_actions":  parsed.get("treatment_actions") or [],
        "preventive_actions": parsed.get("preventive_actions") or [],
        "warnings":           base_warnings + (parsed.get("warnings") or []),
        "raw_llm_output":     raw_llm_text,
    }

    if DEBUG:
        print("\n===== FINAL RESPONSE =====\n")
        print(result)

    return result