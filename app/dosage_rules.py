"""
dosage_rules.py — Dosage rules and treatment products by disease and farming mode.
"""

from typing import Dict, Any, Optional
import re


# ── Dosage rules by disease and farming mode ───────────────────────────────────
DOSAGE_RULES: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "plasmopara_viticola": {
        "conventional": {"dose_l_ha": 1.6, "volume_bouillie_l_ha": 250.0},
        "organic":      {"dose_l_ha": 2.8, "volume_bouillie_l_ha": 300.0},
    },
    "erysiphe_necator": {
        "conventional": {"dose_l_ha": 0.8, "volume_bouillie_l_ha": 220.0},
        "organic":      {"dose_l_ha": 6.0, "volume_bouillie_l_ha": 250.0},
    },
    "colomerus_vitis": {
        "conventional": {"dose_l_ha": 0.9, "volume_bouillie_l_ha": 180.0},
        "organic":      {"dose_l_ha": 1.2, "volume_bouillie_l_ha": 200.0},
    },
    "elsinoe_ampelina": {
        "conventional": {"dose_l_ha": 2.0, "volume_bouillie_l_ha": 250.0},
        "organic":      {"dose_l_ha": 3.0, "volume_bouillie_l_ha": 300.0},
    },
    "guignardia_bidwellii": {
        "conventional": {"dose_l_ha": 1.5, "volume_bouillie_l_ha": 250.0},
        "organic":      {"dose_l_ha": 2.5, "volume_bouillie_l_ha": 300.0},
    },
    "phaeomoniella_chlamydospora": {
        "conventional": {"dose_l_ha": None, "volume_bouillie_l_ha": 200.0},
        "organic":      {"dose_l_ha": None, "volume_bouillie_l_ha": 200.0},
    },
    "healthy": {
        "conventional": {"dose_l_ha": 0.0, "volume_bouillie_l_ha": 0.0},
        "organic":      {"dose_l_ha": 0.0, "volume_bouillie_l_ha": 0.0},
    },
}


# ── Treatment products by disease and farming mode ─────────────────────────────
TREATMENT_PRODUCTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "plasmopara_viticola": {
        "conventional": {
            "type": "Anti-downy mildew (fungicides)",
            "examples": [
                "CAA family (e.g. dimethomorph)",
                "QoI family (e.g. azoxystrobin)",
                "Contact products (e.g. folpet)"
            ],
            "dose_unit": "kg/ha or L/ha (depending on formulation)",
            "strategy": "Preventive + reinforce after rainfall",
            "note": "Alternate fungicide families to limit resistance. Increase frequency during repeated rainfall."
        },
        "organic": {
            "type": "Copper / biocontrol",
            "examples": [
                "Copper (copper hydroxide / Bordeaux mixture)",
                "Phosphonates (subject to local regulations)",
                "Natural defense stimulators (NDS)"
            ],
            "dose_unit": "kg/ha",
            "strategy": "Preventive",
            "note": "Copper is mainly preventive: target risk periods (leaf wetness). Respect annual regulatory limits."
        },
    },
    "erysiphe_necator": {
        "conventional": {
            "type": "Anti-powdery mildew (fungicides)",
            "examples": [
                "Triazoles (e.g. myclobutanil / tebuconazole)",
                "Strobilurins (QoI)",
                "Sulfur (as complement if compatible)"
            ],
            "dose_unit": "kg/ha or L/ha",
            "strategy": "Strict preventive",
            "note": "Powdery mildew cannot be reversed: regularity is critical. Mandatory rotation of modes of action."
        },
        "organic": {
            "type": "Sulfur / biocontrol",
            "examples": [
                "Wettable sulfur",
                "Potassium bicarbonate",
                "Vegetable oils (depending on conditions)"
            ],
            "dose_unit": "kg/ha",
            "strategy": "Preventive",
            "note": "Sulfur is effective but risk of phytotoxicity above 30°C. Adjust interval based on weather and disease pressure."
        },
    },
    "colomerus_vitis": {
        "conventional": {
            "type": "Acaricide",
            "examples": [
                "Abamectin (subject to authorization)",
                "Spirodiclofen (subject to authorization)",
                "Hexythiazox (subject to authorization)"
            ],
            "dose_unit": "L/ha",
            "strategy": "Targeted",
            "note": "Intervene early if outbreak detected. Avoid systematic treatments to preserve beneficial fauna."
        },
        "organic": {
            "type": "Oils / soap / sulfur",
            "examples": [
                "Paraffinic oil (white oils)",
                "Black soap (mechanical effect)",
                "Sulfur (partial effect)"
            ],
            "dose_unit": "L/ha",
            "strategy": "Pressure reduction",
            "note": "Mainly mechanical approach: target the right stage and ensure good coverage. Repeat if necessary."
        },
    },
    "elsinoe_ampelina": {
        "conventional": {
            "type": "Contact fungicide",
            "examples": [
                "Mancozeb (if locally authorized)",
                "Folpet",
                "Copper (depending on strategy)"
            ],
            "dose_unit": "kg/ha",
            "strategy": "Preventive",
            "note": "Intervene early on young tissues during humid periods. Reinforce after heavy rain or rapid growth."
        },
        "organic": {
            "type": "Copper / biocontrol",
            "examples": [
                "Copper (hydroxide / Bordeaux mixture)",
                "Biocontrol (plant extracts subject to authorization)"
            ],
            "dose_unit": "kg/ha",
            "strategy": "Preventive",
            "note": "Efficacy depends on application regularity and weather (rain = washout). Improve ventilation."
        },
    },
    "guignardia_bidwellii": {
        "conventional": {
            "type": "Anti-black rot (fungicides)",
            "examples": [
                "Dithiocarbamates (subject to regulations)",
                "Strobilurins (QoI)",
                "Contact fungicides (e.g. captan / folpet depending on availability)"
            ],
            "dose_unit": "kg/ha or L/ha",
            "strategy": "Preventive + reinforce after rainfall",
            "note": "Target risk periods (rain + heat). Ensure good bunch coverage and renew after washout."
        },
        "organic": {
            "type": "Copper / biocontrol",
            "examples": [
                "Copper (Bordeaux mixture / hydroxide)",
                "Natural defense stimulators (NDS)"
            ],
            "dose_unit": "kg/ha",
            "strategy": "Preventive",
            "note": "Mainly preventive protection. Reinforce prophylaxis (ventilation, removal of infected debris)."
        },
    },
    "phaeomoniella_chlamydospora": {
        "conventional": {
            "type": "No direct curative treatment",
            "examples": [
                "Sanitary pruning / trunk surgery (depending on practice)",
                "Replacement of severely affected vines"
            ],
            "dose_unit": "",
            "strategy": "Prophylaxis + vineyard management",
            "note": "Esca is a wood disease: the approach is mainly agronomic (hygiene, wound protection, vine management)."
        },
        "organic": {
            "type": "No direct curative treatment",
            "examples": [
                "Prophylaxis (pruning hygiene)",
                "Water stress management and canopy ventilation"
            ],
            "dose_unit": "",
            "strategy": "Prophylaxis",
            "note": "Same logic: wood disease. Monitoring + cultural measures; no standard product treatment available."
        },
    },
    "healthy": {
        "conventional": {
            "type": "None",
            "examples": [],
            "dose_unit": "",
            "strategy": "None",
            "note": "No treatment required. Maintain regular monitoring."
        },
        "organic": {
            "type": "None",
            "examples": [],
            "dose_unit": "",
            "strategy": "None",
            "note": "No treatment required. Maintain regular monitoring."
        },
    },
}


# ── Severity multipliers ───────────────────────────────────────────────────────
SEVERITY_MULTIPLIER = {
    "low":      0.75,
    "moderate": 1.0,
    "high":     1.4,
}


# ── CNN label aliases ──────────────────────────────────────────────────────────
CNN_LABEL_ALIASES: Dict[str, str] = {
    "colomerus_vitis":             "colomerus_vitis",
    "elsinoe_ampelina":            "elsinoe_ampelina",
    "erysiphe_necator":            "erysiphe_necator",
    "guignardia_bidwellii":        "guignardia_bidwellii",
    "healthy":                     "healthy",
    "phaeomoniella_chlamydospora": "phaeomoniella_chlamydospora",
    "plasmopara_viticola":         "plasmopara_viticola",
}


# ── Helper functions ───────────────────────────────────────────────────────────

def _normalize_cnn_label(raw_label: str) -> str:
    """
    Normalizes a CNN label to a valid DOSAGE_RULES key.
    Accepts INRAE scientific names and common aliases.
    """
    if not raw_label:
        return raw_label

    label = str(raw_label).strip()
    label = re.sub(r"\.md$", "", label, flags=re.IGNORECASE)
    return CNN_LABEL_ALIASES.get(label.lower(), label)


def format_treatment_product(product: Dict[str, Any]) -> list[str]:
    """
    Converts a treatment product dictionary into a readable list of bullet points.
    """
    if not product:
        return []

    bullets = []
    if product.get("type"):
        bullets.append(f"Product type: {product['type']}")
    if product.get("examples"):
        bullets.append(f"Examples: {', '.join(product['examples'])}")
    if product.get("dose_unit"):
        bullets.append(f"Indicative unit: {product['dose_unit']}")
    if product.get("strategy"):
        bullets.append(f"Strategy: {product['strategy']}")
    if product.get("note"):
        bullets.append(f"Note: {product['note']}")
    return bullets


def compute_dosage(
    cnn_label: str,
    mode: str,
    area_m2: float,
    severity: Optional[str] = None,
    safety_margin: float = 0.10,
) -> Dict[str, Any]:
    """
    Computes treatment volumes based on:
    - disease label (cnn_label)
    - farming mode (conventional or organic)
    - area in m²
    - severity level (low, moderate, high)
    - safety margin (10% by default)

    Returns a dictionary with dosage details and treatment product information.
    """
    cnn_label_norm = _normalize_cnn_label(cnn_label)

    if cnn_label_norm not in DOSAGE_RULES or mode not in DOSAGE_RULES.get(cnn_label_norm, {}):
        return {}

    rules = DOSAGE_RULES[cnn_label_norm][mode]
    dose_l_ha = rules.get("dose_l_ha")
    mult = SEVERITY_MULTIPLIER.get((severity or "").strip().lower(), 1.0)
    dose_l_ha_eff = (dose_l_ha or 0.0) * mult
    volume_bouillie_l_ha = float(rules.get("volume_bouillie_l_ha") or 0.0)
    fraction_ha = area_m2 / 10_000.0

    # Healthy leaf — no treatment needed
    if dose_l_ha == 0.0 and volume_bouillie_l_ha == 0.0:
        return {
            "area_m2": area_m2,
            "dose_l_ha": 0.0,
            "volume_bouillie_l_ha": 0.0,
            "estimated_product_l_for_area": 0.0,
            "estimated_volume_l_for_area": 0.0,
            "configured": True,
            "note": "No treatment required for this disease/severity level.",
        }

    # Disease with no dose configured (e.g. Esca)
    if dose_l_ha is None:
        bouillie_l = volume_bouillie_l_ha * fraction_ha * (1.0 + safety_margin)
        treatment_product = TREATMENT_PRODUCTS.get(cnn_label_norm, {}).get(mode)
        return {
            "area_m2": area_m2,
            "dose_l_ha": None,
            "volume_bouillie_l_ha": volume_bouillie_l_ha,
            "estimated_product_l_for_area": None,
            "estimated_volume_l_for_area": round(bouillie_l, 2),
            "treatment_product": format_treatment_product(treatment_product),
            "configured": False,
            "note": "No dose configured for this label/mode. See treatment product recommendations.",
        }

    # Standard case
    produit_l = dose_l_ha_eff * fraction_ha * (1.0 + safety_margin)
    bouillie_l = volume_bouillie_l_ha * fraction_ha * (1.0 + safety_margin)
    treatment_product = TREATMENT_PRODUCTS.get(cnn_label_norm, {}).get(mode)

    return {
        "area_m2": area_m2,
        "dose_l_ha": round(dose_l_ha_eff, 2),
        "volume_bouillie_l_ha": volume_bouillie_l_ha,
        "estimated_product_l_for_area": round(produit_l, 2),
        "estimated_volume_l_for_area": round(bouillie_l, 2),
        "treatment_product": format_treatment_product(treatment_product),
        "configured": True,
    }