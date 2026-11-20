import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
DB_DIR = BASE_DIR / "qdrant_db"

DOCUMENTS_PATH = DATA_DIR / "documents.json"

COLLECTION_NAME = "company_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 64

SCORE_THRESHOLD = float(
    os.getenv("SCORE_THRESHOLD", "0.35")
)

CONTEXT_LIMIT = int(
    os.getenv("CONTEXT_LIMIT", "4000")
)

RESERVED_OUTPUT_TOKENS = int(
    os.getenv("RESERVED_OUTPUT_TOKENS", "500")
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

NVIDIA_API_KEY= os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL= os.getenv("NVIDIA_MODEL")
NVIDIA_BASE_URL= os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

API_KEY = os.getenv("API_KEY")

FALLBACK = "I don't have information on that"
