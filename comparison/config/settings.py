
"""
Common settings for the RAG comparison test.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# NCloud API Configuration
NCLOUD_API_KEY = os.getenv("NCLOUD_API_KEY")
NCLOUD_APIGW_API_KEY = os.getenv("NCLOUD_APIGW_API_KEY")
NCLOUD_API_URL = os.getenv("NCLOUD_API_URL", "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007")
NCLOUD_REQUEST_ID = os.getenv("NCLOUD_REQUEST_ID", "pageindex-comparison")

# OpenAI API Configuration (for PageIndex Tree Generation)
OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_storage")

# Vector RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 1024  # NCloud Embedding v2

# PageIndex Settings
PAGEINDEX_MODEL = "gpt-4o"
