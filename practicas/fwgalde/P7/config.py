# config.py

# --- Directorios ---
DOCS_PATH   = "./docs"
PERSIST_DIR = "./db/chroma"

# --- Modelos ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"
LLM_TEMPERATURE = 0.2
LLM_NUM_THREAD = 2

# --- Procesamiento de texto ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP =300
SEPARATORS = ["\n\n", "\n", "(?<=\. )", " ", ""]
