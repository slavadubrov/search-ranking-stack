"""
Configuration constants for the search ranking stack.
Paths, model names, and hyperparameters.
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    # Start from the config.py file location
    current = Path(__file__).resolve().parent

    # Walk up until we find pyproject.toml
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:  # Hit filesystem root
            break
        current = parent

    # Fallback: use cwd if we can't find it
    return Path.cwd()


# --- Paths ---
PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ESCI_DIR = DATA_DIR / "esci_sample"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
ESCI_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- ESCI Dataset ---
ESCI_HF_DATASET = "tasksource/esci"
ESCI_LOCALE = "us"
ESCI_TARGET_QUERIES = 500
ESCI_SEED = 42
ESCI_LABEL_MAP = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}

# --- Models ---
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
LLM_MODEL_LOCAL = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL_API = "claude-haiku-4-5-20251001"

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Retrieval Parameters ---
TOP_K_RETRIEVAL = 100  # Number of candidates from retrieval stage
TOP_K_RERANK_CE = 50  # Number of candidates for cross-encoder reranking
TOP_K_RERANK_LLM = 10  # Number of candidates for LLM listwise reranking
RRF_K = 60  # Reciprocal Rank Fusion constant (Cormack et al. 2009)

# Embedding cache
CORPUS_EMBEDDINGS_PATH = ESCI_DIR / "corpus_embeddings.npy"

# --- Evaluation ---
EVAL_METRICS = {"ndcg_cut_10", "recip_rank", "recall_100"}
