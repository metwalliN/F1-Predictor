# Toto — Graph-RAG Formula 1 assistant 

A Streamlit assistant that answers Formula 1 questions from the **Milestone 2 Neo4j knowledge graph** and runs the
**saved Milestone 1 podium model** with a live SHAP explanation. The graph and the model supply the facts; the
language model only turns them into sentences.

```
message ─► entity extraction ─► intent classifier ─► session cache / clarification
              (lookups from the graph)  (rule-based | embedding)        │
                                                                        ├─► out of scope ─► refusal (no trace)
                                                                        ├─► podium_prediction ─► Prediction view
                                                                        └─► template planner ─► Cypher template(s) ─► Neo4j
                                                                                                    │ rows (or "no data")
                                                       two LLMs side by side ◄─ grounded prompt ◄───┘
                                                       (fallback: direct reading of the rows)
Prediction view: pre-race row (Milestone 1 CSV features) ─► saved pipeline ─► probability + class
                 ─► live SHAP ─► + global SHAP + pre-race graph facts (races before this one) ─► LLM explanation
```

## Quick start (talk to it in about 15 minutes)

You need Python 3.12 or 3.13 (not 3.14 yet).

1. Open a terminal in this folder and install the packages (5-10 minutes):
   `python -m pip install -r requirements.txt`
2. Get a free Groq key at https://console.groq.com/keys, copy `.env.example` to `.env` and paste the key after
   `GROQ_API_KEY=`. Without a key the chat still works, but each answer shows "API connection error" and the facts
   are read straight from the graph.
3. Start it: `python start.py` (on Windows you can double-click `start.bat`). The browser opens at
   http://localhost:8501.

If Neo4j isn't running (or `.env` has no `NEO4J_PASSWORD`), `start.py` builds an offline copy of the same graph from
`data/graph_csv/` on the first start (about 20 seconds) and runs the same Cypher templates on it. That copy has the
same 51,113 nodes and 163,510 relationships as the Milestone 2 graph. **For the evaluation, run on Neo4j.**

For the embedding intent classifier (it installs PyTorch, a large download):
`python -m pip install -r requirements-embedding.txt`. Without it the app uses the rule-based classifier.

Models offered in the chat (all free): `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` on Groq, and
`gemma3-4b-local` or `llama3.2-3b-local` through Ollama (no key; `ollama pull gemma3:4b`). Groq retired its free Llama
3.x and Qwen3-32B models in July-August 2026; check https://console.groq.com/docs/models if a model stops answering.

## Run it on Neo4j (the milestone setup)


1. **Python packages**: `pip install -r requirements.txt` and `pip install -r requirements-embedding.txt`.
   `scikit-learn` must be the version recorded in `models/model_card.json`, the one that saved the model.
2. **Secrets**: copy `.env.example` to `.env` and fill in the Neo4j password and at least one LLM key
   (Groq's free tier covers the default models). `.env` is in `.gitignore`; never commit it.
3. **Knowledge graph**: start Neo4j with the Milestone 2 graph loaded (`Create_kg.py`, unchanged).
4. **Model files**: `models/` holds the Milestone 1 export. To regenerate it, run the Milestone 1 notebook
   (`notebook/acl-final-milestone-with-m3-export.ipynb`, cell **M3. Export for Milestone 3**) and copy its
   `m3_artifacts/` folder into `models/`. The app checks on start-up that the loaded model reproduces the notebook's
   1,352 test-set probabilities exactly.
5. **Entity lookups** (once per graph): `python scripts/build_lookups.py`
6. **Start**: `python start.py` (it uses Neo4j when `.env` has the password and the database answers)

The first start downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB). Without it the app falls back to the
rule-based classifier and says so in the sidebar.

## What it answers (intents)

| Intent | Brief row | Entities | Template(s) | Example |
|---|---|---|---|---|
| `driver_stats` | Driver performance | driver, season range (default: career) | `driver_season_stats` | How many podiums did Leclerc get in 2022? |
| `driver_at_circuit` | Driver performance | driver + circuit, season range | `driver_at_circuit` | How many wins did Hamilton have at Silverstone before 2020? |
| `constructor_stats` | Constructor performance | constructor, season range | `constructor_stats` | How reliable was Williams between 2014 and 2021? |
| `head_to_head` | Head-to-head | two drivers, optional constructor | `head_to_head`, `head_to_head_at_constructor` | Who finished ahead more often, Norris or Ricciardo at McLaren? |
| `race_result` | Race and circuit facts | race (Grand Prix + year, or circuit + year) | `race_podium` | Who won the 2019 German Grand Prix? |
| `circuit_info` | Race and circuit facts | circuit | `circuit_info`, `circuit_top_winners` | Where is the Hungaroring? |
| `qualifying_grid` | Qualifying and grid | race, or driver + seasons | `race_grid_gains`, `race_qualifying_top`, `driver_qualifying_record` | Which driver gained the most places from the grid at Monza in 2021? |
| `pit_stops` | Pit stops and strategy | race, or driver + seasons | `race_fastest_pit_stops`, `race_pit_summary`, `driver_pit_stops` | What was the fastest pit stop at the 2023 Qatar Grand Prix? |
| `season_summary` | Season and history | one season, or two for team changes | `season_driver_points`, `season_constructor_points`, `season_overview`, `season_team_changes` | Which drivers changed teams between 2020 and 2021? |
| `podium_prediction` | Prediction path | driver + race, optional grid / qualifying | prediction view; `prerace_driver_at_circuit`, `prerace_constructor_reliability` | Predict Norris at the 2024 Miami Grand Prix |
| `out_of_scope` | Refusal | — | none | Who designed the 2023 Red Bull car? |

Multi-hop (2+ hops): `driver_stats`, `driver_at_circuit`, `constructor_stats`, `circuit_top_winners`,
`season_summary`, `qualifying_grid` (driver), `pit_stops` (driver). Multi-entity: `driver_at_circuit`,
`head_to_head`, `constructor_stats`. Overlapping the five Milestone 2 verification queries: `constructor_stats`
(Q2), `head_to_head` (Q3), `season_summary` (Q1, Q5), and `driver_at_circuit` partly (Q4). New: `driver_stats`,
`race_result`, `circuit_info`, `qualifying_grid`, `pit_stops`, `podium_prediction`, `out_of_scope` (7 of 11).

**Refused questions** (message: "I don't have the knowledge to answer that"): anything outside Formula 1, and F1
topics the graph has no data for — car design and engineers, tyres and compounds, weather, salaries and contracts,
sprint races, lap times and lap records, top speeds, penalties, crashes, news. Out of scope is never decided by
similarity alone: it needs one of those topics, or a message with no Formula 1 word and no recognised entity.
Questions inside the graph's scope that return no rows (e.g. a 2026 season, pit stops before 2011) get "The
knowledge graph has no data for that question." and the language model is not called.

## Statements required by the brief

- **The graph schema is unchanged.** Every template uses only Milestone 2 labels (`Driver`, `Constructor`,
  `Circuit`, `Season`, `Race`, `Result`, `Qualifying`, `PitStop`, `Status`), relationship types and directions, and
  properties. Nothing is written to the graph. `tests/test_templates.py <csv_dir> neo4j` checks every template
  against an independent pandas computation on your graph.
- **Pre-race features** come from `models/prerace_feature_rows.csv`, exported by the Milestone 1 notebook (cells
  FE1–FE4) from the CSVs with the Milestone 1 leakage rules: form over the previous 5 races only (`shift(1)`),
  championship standings from the latest snapshot strictly before the race (season openers: last season's final),
  qualifying and grid known before the start. Standings are not in the graph and the graph is not extended for them.
  The model is never refitted in the app.
- **Pre-race graph facts** used in explanations are restricted to races strictly before the scored race
  (`r.year < $year OR (r.year = $year AND r.round < $round)`).
- **API keys** are read from `.env` / environment variables only.
- **Every turn is logged** to `logs/turns.jsonl`: message, intent decision, entities, cache, templates, Cypher,
  parameters, rows, prompt, answers, model, latency and token counts.

## Repository layout and ownership

| Part (owner) | Files |
|---|---|
| 1 · Intent + entities | `f1rag/preprocessing/intents.py`, `entities.py`, `classifiers.py`; `eval/eval_intents.py`; `eval/testsets/intent_labelled.csv`, `intent_heldout.csv` |
| 2 · Retrieval + prediction | `f1rag/retrieval/templates.py`, `planner.py`; `f1rag/prediction/`; `eval/testsets/qa_testset.jsonl`, `explanation_cases.csv`; error analysis in `eval/run_eval.py` |
| 3 · LLM layer | `f1rag/llm/providers.py`, `prompts.py`, `answer.py`; `eval/metrics.py`, `eval/run_eval.py`, `eval/qualitative_sheet.py` |
| 4 · Session + app | `f1rag/pipeline.py` (cache, follow-ups, clarification, log); `app/streamlit_app.py`; manual-input validation in `predictor.validate` |

## Evaluation

```
python eval/eval_intents.py                        # accuracy + confusion matrix, both classifiers (labelled set)
python eval/eval_intents.py --set intent_heldout   # same on questions never used while writing the rules
python eval/run_eval.py --sleep 2                  # QA pipeline, 3 LLMs x zero/few-shot, OOS refusal, explanations, errors
python eval/run_eval.py --tag after_fix --sleep 2  # rerun after an improvement (before/after)
python eval/qualitative_sheet.py                   # 44 questions x every model -> CSV for two raters
```
Results land in `eval/results/`. The QA set has 40 items (22 normal, 4 ambiguous names, 5 follow-ups,
5 out-of-scope, 4 empty results); its gold facts are the rows the correct template returns at evaluation time.

**A note on the intent numbers.** The rules were written while looking at `intent_labelled.csv`, so they score 100%
there. On `intent_heldout.csv` (48 questions written afterwards and not tuned on) they score 77% — the gap is the
honest measure of how well hand-written paraphrase patterns generalise, and the comparison with the embedding
classifier should be made on the held-out set.

## Offline tests (no Neo4j, no API keys)

```
python tests/build_kuzu_graph.py <csv_dir> /tmp/f1.db        # embedded copy of the Milestone 2 schema
python tests/test_templates.py <csv_dir> /tmp/f1.db          # every template vs pandas
F1_GRAPH_BACKEND=kuzu:/tmp/f1.db INTENT_CLASSIFIER=rule python tests/test_conversation.py /tmp/f1.db
F1_GRAPH_BACKEND=kuzu:/tmp/f1.db python tests/test_app.py    # Streamlit app, headless
```

## Known limitations

- Points are summed Grand Prix race points; sprint points and historical dropped-score rules are not in the graph,
  so season totals can differ from the official tables (the scope note says so in every answer).
- Pit-stop data starts in 2011 and qualifying in 1994; earlier questions return "no data".
- Predictions are available for the 2003–2024 driver-races in the Milestone 1 table; any other race needs manual input.
- A surname shared by several drivers triggers a clarification unless one of them has 10× the race entries
  (Lewis vs Duncan Hamilton); "Verstappen" and "Schumacher" therefore always ask.
- Free-tier model catalogues change; check the names in `f1rag/llm/providers.py` before running the comparison.
