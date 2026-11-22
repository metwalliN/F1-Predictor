"""Settings read from environment variables or a local .env file (never committed)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:          # python-dotenv is optional; plain environment variables also work
    pass


def env(name, default=''):
    return os.environ.get(name, default).strip()


NEO4J_URI = env('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = env('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = env('NEO4J_PASSWORD')
NEO4J_DATABASE = env('NEO4J_DATABASE', 'neo4j')

GROQ_API_KEY = env('GROQ_API_KEY')
OPENROUTER_API_KEY = env('OPENROUTER_API_KEY')
HF_TOKEN = env('HF_TOKEN')
OLLAMA_URL = env('OLLAMA_URL', 'http://localhost:11434')

EMBEDDING_MODEL = env('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

MODELS_DIR = ROOT / 'models'
LOOKUP_DIR = ROOT / 'data' / 'lookups'
LOG_DIR = ROOT / 'logs'
EVAL_DIR = ROOT / 'eval'

# Seasons covered by the knowledge graph (Milestone 2 snapshot)
FIRST_SEASON, LAST_SEASON = 1950, 2024
