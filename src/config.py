from pathlib import Path

PRJ_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_BATCH_SIZE = 16
OLLAMA_BASE_URL = "http://localhost:11434/v1"
