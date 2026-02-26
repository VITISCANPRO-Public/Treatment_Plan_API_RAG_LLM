"""
config.py — Centralized configuration for the Vitiscan Treatment Plan API.
All constants and environment variables are defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Farming modes ──
SUPPORTED_MODES = ("conventional", "organic")
DEFAULT_MODE = "conventional"

# ── Severity levels ──
SUPPORTED_SEVERITIES = ("low", "moderate", "high")
DEFAULT_SEVERITY = "low"

# ── Season ──
DEFAULT_SEASON = "unknown"

# ── Dosage ──
MIN_RECOMMENDED_VOLUME_L_HA = 200
MAX_RECOMMENDED_VOLUME_L_HA = 400

# ── Hugging Face Inference API ──
HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://router.huggingface.co/v1/chat/completions",
)
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)

# ── Weaviate ──
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# ── Knowledge base ──
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge")