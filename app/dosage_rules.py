from typing import Dict, Any

# Règles de dosage par maladie et par mode (conventionnel / bio).
# Les valeurs ci-dessous sont des exemples à remplacer par de vraies données.
DOSAGE_RULES: Dict[str, Dict[str, Dict[str, float]]] = {
    "Grape_Downy_mildew_leaf": {
        "conventionnel": {
            "dose_l_ha": 2.0,          # litres de produit pur / ha
            "volume_bouillie_l_ha": 200.0  # litres de bouillie totale / ha
        },
        "bio": {
            "dose_l_ha": 3.0,
            "volume_bouillie_l_ha": 200.0
        },
    },
    # TODO: ajouter les autres maladies ici...
}


def compute_dosage(
    cnn_label: str,
    mode: str,
    area_m2: float,
    safety_margin: float = 0.10,
) -> Dict[str, Any]:
    """
    Calcule les volumes à préparer à partir :
    - d'un label de maladie (cnn_label),
    - d'un mode (bio ou conventionnel),
    - d'une surface en m²,
    - d'une marge de sécurité (10 % par défaut).
    """

    if cnn_label not in DOSAGE_RULES or mode not in DOSAGE_RULES[cnn_label]:
        # Si on ne trouve pas de règle, on renvoie un dict vide.
        return {}

    rules = DOSAGE_RULES[cnn_label][mode]

    dose_l_ha = rules["dose_l_ha"]
    volume_bouillie_l_ha = rules["volume_bouillie_l_ha"]

    # Conversion m² -> fraction d'hectare.
    fraction_ha = area_m2 / 10_000.0

    # Calculs bruts.
    produit_l = dose_l_ha * fraction_ha
    bouillie_l = volume_bouillie_l_ha * fraction_ha

    # Application de la marge de sécurité (ex: +10 %).
    produit_l *= 1.0 + safety_margin
    bouillie_l *= 1.0 + safety_margin

    return {
        "area_m2": area_m2,
        "dose_l_ha": dose_l_ha,
        "volume_bouillie_l_ha": volume_bouillie_l_ha,
        "estimated_product_l_for_area": round(produit_l, 2),
        "estimated_volume_l_for_area": round(bouillie_l, 2),
    }
