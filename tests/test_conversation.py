"""Scripted conversations through the whole pipeline (no LLM calls): follow-ups, cache, clarification, refusal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1rag.graph.client import KuzuClient                        # noqa: E402
from f1rag.pipeline import Assistant, SessionState               # noqa: E402
from f1rag.preprocessing.classifiers import RuleClassifier       # noqa: E402

client = KuzuClient(sys.argv[1])
bot = Assistant(client, RuleClassifier())

SCRIPTS = [
    ['How many podiums did Leclerc get in 2022?', 'And in 2020?', 'What about Norris?', '2021?',
     'Who designed the 2023 Red Bull car?', 'How many points did he score in 2019?'],
    ['Who finished ahead more often, Norris or Ricciardo at McLaren?', 'What about Piastri?'],
    ['How many points did Schumacher score in 2000?', 'Michael'],
    ['Who won the German Grand Prix?', '2019', 'And in 2018?'],
    ['How many wins did Hamilton have at Silverstone before 2020?', 'And at Monza?'],
    ['Who won at Silverstone in 2020?', '70th Anniversary'],
    ['Which driver gained the most places from the grid at Monza in 2021?'],
    ['What was the fastest pit stop at the 2023 Qatar Grand Prix?', 'And in 2010?'],
    ['Which drivers changed teams between 2020 and 2021?'],
    ['How reliable was Williams between 2014 and 2021?'],
    ['Where is the Hungaroring?'],
    ['Predict Norris at the 2024 Miami Grand Prix starting from P3'],
    ['What is the capital of France?'],
    ['How many wins did Leclerc get in 2026?'],
]
for script in SCRIPTS:
    st = SessionState()
    print('=' * 100)
    for msg in script:
        r = bot.handle(msg, st)
        rows = r.retrieval.rows[:2] if r.retrieval else ''
        print(f'> {msg}\n  [{r.kind}] intent={r.intent} status={(r.decision or {}).get("status")} '
              f'templates={[c["template"] for c in r.retrieval.calls] if r.retrieval else ""}\n'
              f'  slots={r.slots}\n  msg={r.message[:160]!r}\n  rows={rows}\n  cache={st.cache_view()}')
