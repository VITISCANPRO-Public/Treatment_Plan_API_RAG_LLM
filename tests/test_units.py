"""
test_units.py — Unit tests for pure functions.

These tests require NO external services (no Weaviate, no HuggingFace, no S3).
They test pure Python logic that can be verified offline.

Covered modules:
  - app.rag_pipeline : infer_season_from_date, parse_llm_structured_response
  - app.dosage_rules : compute_dosage, _normalize_cnn_label
"""

import pytest
from app.rag_pipeline import infer_season_from_date, parse_llm_structured_response
from app.dosage_rules import compute_dosage, _normalize_cnn_label


# ═══════════════════════════════════════════════════════════════════════════════
# infer_season_from_date
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferSeasonFromDate:
    """Tests for the infer_season_from_date() helper."""

    def test_spring_march(self):
        assert infer_season_from_date("2024-03-15") == "spring"

    def test_spring_april(self):
        assert infer_season_from_date("2024-04-01") == "spring"

    def test_spring_may(self):
        assert infer_season_from_date("2024-05-31") == "spring"

    def test_summer_june(self):
        assert infer_season_from_date("2024-06-01") == "summer"

    def test_summer_july(self):
        assert infer_season_from_date("2024-07-15") == "summer"

    def test_summer_august(self):
        assert infer_season_from_date("2024-08-31") == "summer"

    def test_autumn_september(self):
        assert infer_season_from_date("2024-09-01") == "autumn"

    def test_autumn_october(self):
        assert infer_season_from_date("2024-10-20") == "autumn"

    def test_autumn_november(self):
        assert infer_season_from_date("2024-11-30") == "autumn"

    def test_winter_december(self):
        assert infer_season_from_date("2024-12-01") == "winter"

    def test_winter_january(self):
        assert infer_season_from_date("2024-01-15") == "winter"

    def test_winter_february(self):
        assert infer_season_from_date("2024-02-28") == "winter"

    def test_empty_string_returns_unknown(self):
        assert infer_season_from_date("") == "unknown"

    def test_none_returns_unknown(self):
        assert infer_season_from_date(None) == "unknown"

    def test_invalid_date_returns_unknown(self):
        assert infer_season_from_date("not-a-date") == "unknown"

    def test_partial_date_returns_unknown(self):
        assert infer_season_from_date("2024-13-01") == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# _normalize_cnn_label
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeCnnLabel:
    """Tests for the _normalize_cnn_label() helper in dosage_rules."""

    def test_known_label_unchanged(self):
        assert _normalize_cnn_label("plasmopara_viticola") == "plasmopara_viticola"

    def test_all_7_known_labels(self):
        known = [
            "colomerus_vitis",
            "elsinoe_ampelina",
            "erysiphe_necator",
            "guignardia_bidwellii",
            "healthy",
            "phaeomoniella_chlamydospora",
            "plasmopara_viticola",
        ]
        for label in known:
            assert _normalize_cnn_label(label) == label

    def test_strips_whitespace(self):
        assert _normalize_cnn_label("  plasmopara_viticola  ") == "plasmopara_viticola"

    def test_strips_md_extension(self):
        # ingestion.py reads filenames — .md suffix must be cleaned
        assert _normalize_cnn_label("plasmopara_viticola.md") == "plasmopara_viticola"

    def test_unknown_label_returned_as_is(self):
        # Unknown labels should pass through (not raise an exception)
        assert _normalize_cnn_label("unknown_disease") == "unknown_disease"

    def test_empty_string(self):
        result = _normalize_cnn_label("")
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# compute_dosage
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeDosage:
    """Tests for the compute_dosage() function in dosage_rules."""

    def test_healthy_returns_zero_volumes(self):
        result = compute_dosage("healthy", "conventional", 1000.0, severity="low")
        assert result["estimated_product_l_for_area"] == 0.0
        assert result["estimated_volume_l_for_area"] == 0.0

    def test_healthy_returns_configured_true(self):
        result = compute_dosage("healthy", "conventional", 1000.0)
        assert result["configured"] is True

    def test_known_disease_conventional_returns_non_zero(self):
        result = compute_dosage("plasmopara_viticola", "conventional", 10000.0, severity="moderate")
        assert result["estimated_product_l_for_area"] > 0
        assert result["estimated_volume_l_for_area"] > 0

    def test_known_disease_organic_returns_non_zero(self):
        result = compute_dosage("erysiphe_necator", "organic", 5000.0, severity="moderate")
        assert result["estimated_product_l_for_area"] > 0

    def test_area_proportionality(self):
        """Doubling the area should double the volumes (linear relationship)."""
        r1 = compute_dosage("plasmopara_viticola", "conventional", 1000.0, severity="moderate")
        r2 = compute_dosage("plasmopara_viticola", "conventional", 2000.0, severity="moderate")
        assert abs(r2["estimated_product_l_for_area"] - 2 * r1["estimated_product_l_for_area"]) < 0.02

    def test_severity_high_greater_than_low(self):
        """High severity must produce a higher product volume than low severity."""
        r_low  = compute_dosage("plasmopara_viticola", "conventional", 10000.0, severity="low")
        r_high = compute_dosage("plasmopara_viticola", "conventional", 10000.0, severity="high")
        assert r_high["estimated_product_l_for_area"] > r_low["estimated_product_l_for_area"]

    def test_esca_has_no_dose_configured(self):
        """Esca (phaeomoniella) has no chemical dose — configured must be False."""
        result = compute_dosage("phaeomoniella_chlamydospora", "conventional", 1000.0)
        assert result["configured"] is False
        assert result["dose_l_ha"] is None

    def test_esca_still_returns_volume(self):
        """Even without a dose, Esca should return a spray volume estimate."""
        result = compute_dosage("phaeomoniella_chlamydospora", "conventional", 10000.0)
        assert result["estimated_volume_l_for_area"] > 0

    def test_unknown_label_returns_empty_dict(self):
        result = compute_dosage("unknown_disease", "conventional", 1000.0)
        assert result == {}

    def test_unknown_mode_returns_empty_dict(self):
        result = compute_dosage("plasmopara_viticola", "biodynamic", 1000.0)
        assert result == {}

    def test_result_contains_area(self):
        result = compute_dosage("erysiphe_necator", "conventional", 2500.0)
        assert result["area_m2"] == 2500.0

    def test_all_7_diseases_conventional_dont_crash(self):
        diseases = [
            "colomerus_vitis", "elsinoe_ampelina", "erysiphe_necator",
            "guignardia_bidwellii", "healthy", "phaeomoniella_chlamydospora",
            "plasmopara_viticola",
        ]
        for disease in diseases:
            result = compute_dosage(disease, "conventional", 1000.0, severity="moderate")
            assert isinstance(result, dict), f"compute_dosage crashed for {disease}"


# ═══════════════════════════════════════════════════════════════════════════════
# parse_llm_structured_response
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseLLMStructuredResponse:
    """
    Tests for parse_llm_structured_response() in rag_pipeline.

    This function is critical: it converts raw LLM text output
    (which can be malformed) into a clean structured dict.
    """

    VALID_JSON_RESPONSE = '''{
        "diagnostic": "Downy mildew detected on leaves.",
        "treatment_actions": ["Apply copper fungicide.", "Remove infected leaves."],
        "preventive_actions": ["Improve ventilation.", "Monitor weekly."],
        "warnings": ["Respect pre-harvest intervals."]
    }'''

    def test_valid_json_returns_correct_keys(self):
        result = parse_llm_structured_response(self.VALID_JSON_RESPONSE)
        assert set(result.keys()) == {"diagnostic", "treatment_actions", "preventive_actions", "warnings"}

    def test_valid_json_diagnostic_is_string(self):
        result = parse_llm_structured_response(self.VALID_JSON_RESPONSE)
        assert isinstance(result["diagnostic"], str)
        assert len(result["diagnostic"]) > 0

    def test_valid_json_lists_are_lists(self):
        result = parse_llm_structured_response(self.VALID_JSON_RESPONSE)
        assert isinstance(result["treatment_actions"],  list)
        assert isinstance(result["preventive_actions"], list)
        assert isinstance(result["warnings"],           list)

    def test_valid_json_lists_contain_strings(self):
        result = parse_llm_structured_response(self.VALID_JSON_RESPONSE)
        for item in result["treatment_actions"]:
            assert isinstance(item, str)

    def test_json_wrapped_in_code_fences(self):
        """LLMs often wrap JSON in ```json ... ``` — parser must handle this."""
        raw = f"```json\n{self.VALID_JSON_RESPONSE}\n```"
        result = parse_llm_structured_response(raw)
        assert result["diagnostic"] == "Downy mildew detected on leaves."

    def test_json_with_trailing_commas(self):
        """Common LLM mistake: trailing comma in lists."""
        raw = '''{
            "diagnostic": "Test.",
            "treatment_actions": ["Action 1.", "Action 2.",],
            "preventive_actions": [],
            "warnings": []
        }'''
        result = parse_llm_structured_response(raw)
        # Parser should not crash — may return empty or partial result
        assert isinstance(result, dict)
        assert "diagnostic" in result

    def test_empty_string_returns_default_structure(self):
        result = parse_llm_structured_response("")
        assert "diagnostic"        in result
        assert "treatment_actions"  in result
        assert "preventive_actions" in result
        assert "warnings"           in result

    def test_none_input_returns_default_structure(self):
        result = parse_llm_structured_response(None)
        assert isinstance(result, dict)

    def test_plain_text_no_json_returns_text_as_diagnostic(self):
        """If LLM returns plain text, it should end up in diagnostic."""
        raw = "The vine appears to have downy mildew. Apply copper immediately."
        result = parse_llm_structured_response(raw)
        assert isinstance(result, dict)
        assert "diagnostic" in result

    def test_empty_lists_are_valid(self):
        raw = '''{
            "diagnostic": "Healthy leaf detected.",
            "treatment_actions": [],
            "preventive_actions": [],
            "warnings": []
        }'''
        result = parse_llm_structured_response(raw)
        assert result["treatment_actions"]  == []
        assert result["preventive_actions"] == []