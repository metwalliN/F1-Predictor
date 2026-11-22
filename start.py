"""One command to start the assistant:  python start.py   (Windows: double-click start.bat)

- Uses your Neo4j graph when .env has a password and the database answers.
- Otherwise builds an offline copy of the same graph from data/graph_csv (once, about 20 seconds) and uses that,
  so you can talk to the assistant without Neo4j. The milestone itself must run on Neo4j.
- Uses the embedding intent classifier when sentence-transformers is installed, otherwise the rule-based one.
- Without an LLM key the chat still answers, reading the facts straight from the graph.

options:  --offline   always use the offline copy, even if Neo4j is running
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REQUIRED = {'pandas': 'pandas', 'numpy': 'numpy', 'sklearn': 'scikit-learn', 'joblib': 'joblib', 'shap': 'shap',
            'lime': 'lime', 'matplotlib': 'matplotlib', 'streamlit': 'streamlit', 'requests': 'requests',
            'neo4j': 'neo4j', 'dotenv': 'python-dotenv'}


def say(msg):
    print(f'  {msg}')


def main():
    print('\nToto, the F1 Graph-RAG assistant\n')
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        say('Missing packages: ' + ', '.join(missing))
        say(f'Run:  "{sys.executable}" -m pip install -r requirements.txt')
        return 1

    from f1rag import config                     # loads .env
    env = dict(os.environ)

    # ------------------------------------------------------------------ graph
    backend = None
    if '--offline' not in sys.argv and config.NEO4J_PASSWORD:
        try:
            from f1rag.graph.client import Neo4jClient
            c = Neo4jClient()
            n = c.run('MATCH (n) RETURN count(n) AS nodes')[0]['nodes']
            c.close()
            backend = 'neo4j'
            say(f'Graph: Neo4j at {config.NEO4J_URI} ({n:,} nodes)')
        except Exception as e:
            say(f'Neo4j is not reachable ({type(e).__name__}); using the offline copy instead.')
    if backend is None:
        if importlib.util.find_spec('kuzu') is None:
            say('The offline graph needs the kuzu package (Python 3.12 or 3.13):  pip install kuzu==0.11.3')
            say('Or start Neo4j with your Milestone 2 graph and put NEO4J_PASSWORD in .env.')
            return 1
        db = ROOT / 'data' / 'offline_graph.db'
        done = ROOT / 'data' / 'offline_graph.ok'          # written only after a complete build
        if not (db.exists() and done.exists()):
            done.unlink(missing_ok=True)
            say('Building the offline copy of the graph (first start only)...')
            from f1rag.graph.offline_graph import build
            counts, rels = build(ROOT / 'data' / 'graph_csv', db)
            done.write_text(f'{sum(counts.values())} nodes, {rels} relationships\n', encoding='utf-8')
            say(f'Built: {sum(counts.values()):,} nodes, {rels:,} relationships (same as the Milestone 2 graph).')
        env['F1_GRAPH_BACKEND'] = f'kuzu:{db}'
        say('Graph: offline copy (data/offline_graph.db)')

    # ------------------------------------------------------------------ intent classifier
    if config.env('INTENT_CLASSIFIER', 'embedding') == 'embedding' and \
            importlib.util.find_spec('sentence_transformers') is None:
        env['INTENT_CLASSIFIER'] = 'rule'
        say('Intent classifier: rule-based (for the embedding one: pip install -r requirements-embedding.txt)')
    else:
        say(f"Intent classifier: {env.get('INTENT_CLASSIFIER', config.env('INTENT_CLASSIFIER', 'embedding'))}")

    # ------------------------------------------------------------------ language models
    keys = [n for n, v in (('Groq', config.GROQ_API_KEY), ('OpenRouter', config.OPENROUTER_API_KEY),
                           ('Hugging Face', config.HF_TOKEN)) if v]
    if keys:
        say('LLM keys found: ' + ', '.join(keys))
    else:
        say('No LLM key in .env: the chat will show "API connection error" and read the facts from the graph.')
        say('Free key: https://console.groq.com/keys  then add  GROQ_API_KEY=...  to the .env file.')

    say('Opening http://localhost:8501  (Ctrl+C here to stop)\n')
    return subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app' / 'streamlit_app.py'),
                            '--browser.gatherUsageStats', 'false'], env=env, cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
