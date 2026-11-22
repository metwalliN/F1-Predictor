"""Qualitative rating sheet: every compared model's answer to 4 questions per intent. The script prints the items and
writes a CSV with empty columns for two raters (fluency 1-5) and a causal-language flag; the scores are filled in
by hand. An automatic causal-language hint is included to help the raters, not to replace them.

Fluency rubric: 5 natural and precise · 4 clear, minor awkwardness · 3 understandable but clumsy or padded ·
2 hard to follow or partly off-topic · 1 unusable.

usage: python eval/qualitative_sheet.py [--models ...] [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))
from f1rag import config                                                         # noqa: E402
from f1rag.graph.client import get_client                                        # noqa: E402
from f1rag.pipeline import Assistant, SessionState                               # noqa: E402
from f1rag.preprocessing.entities import EntityExtractor                         # noqa: E402
from f1rag.llm.providers import COMPARED                                         # noqa: E402
from f1rag.llm.answer import read_rows                                           # noqa: E402
from f1rag.prediction.predictor import PodiumPredictor                           # noqa: E402
from f1rag.prediction.explain import graph_facts, explain                        # noqa: E402
from run_eval import classifier                                                  # noqa: E402
import metrics as M                                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=COMPARED)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--classifier', default=config.env('INTENT_CLASSIFIER', 'embedding'))
    args = ap.parse_args()
    qs = pd.read_csv(ROOT / 'eval' / 'testsets' / 'qualitative_questions.csv')
    assert qs.groupby('intent').size().eq(4).all()
    cases = pd.read_csv(ROOT / 'eval' / 'testsets' / 'explanation_cases.csv').set_index('id')
    client, extractor = get_client(), EntityExtractor()
    bot = Assistant(client, classifier(extractor, args.classifier), extractor)
    pred = PodiumPredictor()
    models = ['row-reading (dry run)'] if args.dry_run else args.models
    out = []
    for i, q in enumerate(qs.itertuples(), 1):
        if q.intent == 'podium_prediction':
            c = cases.loc[q.question]
            row = pred.lookup(int(c.driverId), int(c.raceId))
            p = pred.predict({f: float(row[f]) for f in pred.features})
            facts, _ = graph_facts(client, int(c.driverId), int(row['constructorId']), int(row['circuitId']),
                                   int(row['year']), int(row['round']))
            question = f'Explain the podium prediction for {c.driver} at the {c.race}'
            for m in models:
                text = pred.fallback_explanation(p) if args.dry_run else \
                    explain(pred, p, c.driver, c.race, facts, m)['text']
                out.append((i, q.intent, question, m, text))
            continue
        res = bot.handle(q.question, SessionState(), models=() if args.dry_run else tuple(models))
        for m in models:
            if res.answers:
                text = next(a.text for a in res.answers if a.model_id == m)
            elif res.kind == 'answer':
                text = read_rows(res.retrieval.calls)
            else:
                text = res.message
            out.append((i, q.intent, q.question, m, text))
    sheet = pd.DataFrame(out, columns=['item', 'intent', 'question', 'model', 'answer'])
    sheet['rater1_fluency_1to5'] = ''
    sheet['rater2_fluency_1to5'] = ''
    sheet['causal_language_flag_Y_N'] = ''
    sheet['auto_causal_hint'] = sheet.answer.map(lambda t: 'check' if M.causal_language(t) else '')
    path = ROOT / 'eval' / 'results' / 'qualitative_sheet.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    for r in sheet.itertuples():
        print(f'[{r.item:02d} {r.intent} | {r.model}] Q: {r.question}\n    A: {r.answer}\n')
    print(f'{len(sheet)} items ({sheet.item.nunique()} questions x {len(models)} models) -> {path}')


if __name__ == '__main__':
    main()
