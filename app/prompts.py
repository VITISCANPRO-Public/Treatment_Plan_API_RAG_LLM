"""
prompts.py — LLM prompt construction for treatment plan generation.
"""

from typing import List, Dict


def build_treatment_prompt(
    cnn_label: str,
    disease_name: str,
    mode: str,
    severity: str,
    area_m2: float,
    season: str,
    context_chunks: List[Dict[str, str]],
) -> str:
    """
    Builds the prompt sent to the LLM to generate a viticultural action plan.

    Expected response: a strictly valid JSON object with 4 fields:
    - diagnostic: str
    - treatment_actions: List[str]
    - preventive_actions: List[str]
    - warnings: List[str]

    Args:
        cnn_label: INRAE scientific name predicted by the CNN
        disease_name: Human-readable disease name in English
        mode: Farming mode ('conventional' or 'organic')
        severity: Severity level ('low', 'moderate', 'high')
        area_m2: Affected area in m²
        season: Inferred season from date
        context_chunks: Relevant knowledge base chunks retrieved from Weaviate

    Returns:
        Formatted prompt string ready to be sent to the LLM
    """
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are an expert viticulture specialist in grapevine disease management.

Situation context:
- Detected disease (CNN label): "{cnn_label}"
- Disease name: "{disease_name}"
- Farming mode: "{mode}"
- Severity: "{severity}"
- Affected area: {area_m2} m²
- Season: "{season}"

Knowledge base (extracts from technical disease sheets):
{context}

Your task:
1) Provide a concise DIAGNOSTIC of the situation.
2) Propose concrete and actionable TREATMENT ACTIONS.
3) Propose PREVENTIVE MEASURES for the rest of the season.
4) List relevant WARNINGS about safety, regulations or pre-harvest intervals.

CRITICAL CONSTRAINT:
You must respond with a SINGLE valid JSON object, with NO text before or after, NO explanation.
Use exactly the following keys:

{{
  "diagnostic": "Short text (3 to 5 sentences max) explaining the situation.",
  "treatment_actions": [
    "Treatment action 1 (specific, operational).",
    "Treatment action 2."
  ],
  "preventive_actions": [
    "Preventive action 1.",
    "Preventive action 2."
  ],
  "warnings": [
    "Warning 1 (safety, regulations, pre-harvest intervals, etc.).",
    "Warning 2."
  ]
}}

Rules:
- Write in English.
- Be concrete, clear and operational for a wine grower.
- Do not talk about yourself, do not apologize, do not thank.
- Do NOT wrap the JSON in ```json or any code block.
- Do NOT add a trailing comma after the last element of a list.
- Do NOT include ANY text outside the JSON object.

Format rules (MANDATORY):
- "treatment_actions", "preventive_actions" and "warnings" must be JSON arrays (List[str]) only.
- Never represent a list as an object with keys "0:", "1:", etc.
- Never use Markdown lists (e.g. "- ...", "* ...", "1) ...").
- Each list item must be a plain string with no numbering prefix.
- Each item should be 1 to 2 sentences max, avoid "Action: ..." key/value format.
- Aim for 2 to 6 items per list.
- Before responding, verify that your JSON would pass json.loads().

Respond now with the JSON object only.
"""
    return prompt.strip()