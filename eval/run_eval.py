"""Full evaluation: pipeline on the QA test set, LLM comparison (zero-shot vs few-shot, same prompts and same
retrieved context for every model), out-of-scope refusal, explanation fidelity, and error analysis by stage.

usage:
  python eval/run_eval.py                                  # COMPARED models from f1rag/llm/providers.py
  python eval/run_eval.py --models gpt-oss-120b gpt-oss-20b qwen3.8-27b
  python eval/run_eval.py --dry-run                        # no API calls: the row reading stands in for answers
  python eval/run_eval.py --tag after_fix                  # keep a second run next to the first (before/after)
Results go to eval/results/<tag>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))
from f1rag import config                                                       # noqa: E402
from f1rag.graph.client import get_client                                      # noqa: E402
from f1rag.pipeline import Assistant, SessionState                             # noqa: E402
from f1rag.preprocessing.entities import EntityExtractor                       # noqa: E402
from f1rag.preprocessing.classifiers import RuleClassifier, EmbeddingClassifier, SentenceTransformerEncoder  # noqa
from f1rag.retrieval.templates import T                                        # noqa: E402
from f1rag.retrieval.planner import _tidy                                      # noqa: E402
from f1rag.llm.answer import answer as llm_answer, read_rows                   # noqa: E402
from f1rag.llm.providers import COMPARED                                       # noqa: E402
from f1rag.prediction.predictor import PodiumPredictor, PLAIN                  # noqa: E402
from f1rag.prediction.explain import graph_facts, explain                      # noqa: E402
import metrics as M                                                            # noqa: E402


def classifier(extractor, kind):
    if kind == 'rule':
        return RuleClassifier()
    return EmbeddingClassifier(SentenceTransformerEncoder(config.EMBEDDING_MODEL), extractor)


def gold_rows(client, extractor, gold):
    params = dict(gold['params'])
    if 'race' in params:
        y, name = params.pop('race')
        params['raceId'] = extractor.race_from_name(name, y).id
    t = T[gold['template']]
    return [_tidy(r) for r in client.run(t.cypher, {**t.fixed_params, **params})], params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=COMPARED)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--classifier', default=config.env('INTENT_CLASSIFIER', 'embedding'))
    ap.add_argument('--tag', default='run1')
    ap.add_argument('--sleep', type=float, default=0.0, help='pause between API calls (free-tier rate limits)')
    args = ap.parse_args()
    out = ROOT / 'eval' / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)

    client, extractor = get_client(), EntityExtractor()
    bot = Assistant(client, classifier(extractor, args.classifier), extractor)
    items = [json.loads(l) for l in (ROOT / 'eval' / 'testsets' / 'qa_testset.jsonl').read_text(encoding='utf-8').splitlines() if l]
    assert len(items) >= 30

    # ---------------------------------------------------------------- 1. pipeline (no LLM)
    pipe = []
    for it in items:
        st = SessionState()
        for msg in it['turns']:
            res = bot.handle(msg, st, models=())
        row = {'id': it['id'], 'category': it['category'], 'question': ' / '.join(it['turns']),
               'expected_kind': it['expect'], 'kind': res.kind, 'expected_intent': it['intent'],
               'intent': res.intent, 'entities': json.dumps(res.slots, ensure_ascii=False),
               'templates': ','.join(c['template'] for c in res.retrieval.calls) if res.retrieval else '',
               'rows': res.retrieval.rows if res.retrieval else [], 'calls': res.retrieval.calls if res.retrieval
               else [], 'slots': res.slots or {}}
        if it.get('gold'):
            g, gp = gold_rows(client, extractor, it['gold'])
            row['gold_template'], row['gold_rows'] = it['gold']['template'], g
            got = next((c for c in row['calls'] if c['template'] == it['gold']['template']), None)
            row['retrieval_ok'] = bool(got) and got['rows'] == g
        pipe.append(row)
    P = pd.DataFrame(pipe)
    P['kind_ok'] = P.kind == P.expected_kind
    P['intent_ok'] = (P.intent == P.expected_intent) | ((P.expected_intent == 'out_of_scope') & (P.kind == 'refuse'))
    P.drop(columns=['calls']).to_csv(out / 'qa_pipeline.csv', index=False)
    oos = P[P.category == 'out_of_scope']
    print(f'Pipeline: kind correct {P.kind_ok.mean():.3f}, intent correct {P.intent_ok.mean():.3f}, '
          f"retrieval matches gold {P.retrieval_ok.dropna().astype(bool).mean():.3f}, "
          f'out-of-scope refused {(oos.kind == "refuse").mean():.3f} ({len(oos)} items)')

    # ---------------------------------------------------------------- 2. LLM comparison
    answerable = [p for p in pipe if p['kind'] == 'answer' and p.get('gold_rows')]
    item_by_id = {it['id']: it for it in items}
    recs = []
    for model in ([] if args.dry_run else args.models) + (['row-reading (dry run)'] if args.dry_run else []):
        for shots in (False, True):
            for p in answerable:
                it = item_by_id[p['id']]
                q = it['turns'][-1]
                if args.dry_run:
                    text, ok, lat, tin, tout = read_rows(p['calls']), True, 0.0, 0, 0
                else:
                    a = llm_answer(model, q, p['intent'], p['slots'], p['calls'], few_shot=shots)
                    text, ok, lat, tin, tout = a.text, a.ok, a.latency_s, a.input_tokens, a.output_tokens
                    time.sleep(args.sleep)
                corr, missing = M.correctness(text, p['gold_rows'], it['answer_fields'])
                faithful, xn, xnames = M.faithfulness(text, p['rows'], ' '.join(it['turns']), p['slots'],
                                                       ' '.join(c['scope'] for c in p['calls']))
                recs.append({'model': model, 'prompting': 'few-shot' if shots else 'zero-shot', 'id': p['id'],
                             'question': q, 'answer': text, 'call_ok': ok, 'correctness': corr,
                             'missing': json.dumps(missing, default=str), 'faithful': faithful,
                             'extra_numbers': ' '.join(xn), 'extra_names': ' '.join(xnames),
                             'causal_language': M.causal_language(text), 'latency_s': lat, 'input_tokens': tin,
                             'output_tokens': tout})
            # out-of-scope probe: the question with the nearest retrieved context; the model should decline
            for it in items:
                if it.get('probe'):
                    rows, _ = gold_rows(client, extractor, it['probe'])
                    calls = [{'template': it['probe']['template'], 'description': T[it['probe']['template']].description,
                              'scope': T[it['probe']['template']].scope, 'rows': rows}]
                    if args.dry_run:
                        text, ok, lat, tin, tout = 'The data does not cover that.', True, 0, 0, 0
                    else:
                        a = llm_answer(model, it['turns'][0], 'probe', {}, calls, few_shot=shots)
                        text, ok, lat, tin, tout = a.text, a.ok, a.latency_s, a.input_tokens, a.output_tokens
                        time.sleep(args.sleep)
                    recs.append({'model': model, 'prompting': 'few-shot' if shots else 'zero-shot', 'id': it['id'],
                                 'question': it['turns'][0], 'answer': text, 'call_ok': ok, 'probe_refused':
                                 M.is_refusal(text), 'latency_s': lat, 'input_tokens': tin, 'output_tokens': tout})
            if args.dry_run:
                break
    L = pd.DataFrame(recs)
    L.to_csv(out / 'llm_answers.csv', index=False)
    if len(L):
        ans = L[L.correctness.notna()] if 'correctness' in L else L
        probe = L[L.get('probe_refused').notna()] if 'probe_refused' in L else L.iloc[0:0]
        summary = ans.groupby(['model', 'prompting']).agg(
            items=('id', 'size'), calls_ok=('call_ok', 'mean'), correctness=('correctness', 'mean'),
            faithfulness=('faithful', 'mean'), causal_language=('causal_language', 'mean'),
            mean_latency_s=('latency_s', 'mean'), mean_input_tokens=('input_tokens', 'mean'),
            mean_output_tokens=('output_tokens', 'mean')).round(3)
        if len(probe):
            summary = summary.join(probe.groupby(['model', 'prompting']).probe_refused.mean().rename(
                'oos_probe_refusal').round(3))
        summary['pipeline_oos_refusal'] = round((oos.kind == 'refuse').mean(), 3)
        summary.to_csv(out / 'llm_summary.csv')
        print('\nLLM comparison (same prompts, same retrieved rows):')
        print(summary.to_string())

    # ---------------------------------------------------------------- 3. explanation fidelity
    pred = PodiumPredictor()
    cases = pd.read_csv(ROOT / 'eval' / 'testsets' / 'explanation_cases.csv')
    erecs = []
    for c in cases.itertuples():
        row = pred.lookup(c.driverId, c.raceId)
        clean = {f: row[f] for f in pred.features}
        p = pred.predict(pd.Series(clean).astype(float).to_dict(), actual_finish=row['actual_finish'])
        facts, _ = graph_facts(client, c.driverId, int(row['constructorId']), int(row['circuitId']), int(row['year']),
                               int(row['round']))
        up, down = pred.top_contributions(p)
        listed = [f for f, _ in up + down]
        for model in (['fallback (dry run)'] if args.dry_run else args.models):
            if args.dry_run:
                e = {'ok': True, 'text': pred.fallback_explanation(p), 'latency_s': 0, 'input_tokens': 0,
                     'output_tokens': 0}
            else:
                e = explain(pred, p, c.driver, c.race, facts, model)
                time.sleep(args.sleep)
            fid = M.explanation_fidelity(e['text'], p.contributions, PLAIN, listed)
            erecs.append({'case': c.id, 'driver': c.driver, 'race': c.race, 'probability': p.probability,
                          'predicted': 'podium' if p.podium else 'not podium', 'actual_finish': p.actual_finish,
                          'model': model, 'call_ok': e['ok'], 'explanation': e['text'],
                          'fidelity_recall_top3': fid['recall_top'], 'fidelity_precision': fid['precision'],
                          'top3': ','.join(fid['top']), 'mentioned': ','.join(fid['mentioned']),
                          'causal_language': M.causal_language(e['text']), 'latency_s': e.get('latency_s', 0),
                          'input_tokens': e.get('input_tokens', 0), 'output_tokens': e.get('output_tokens', 0)})
    E = pd.DataFrame(erecs)
    E.to_csv(out / 'explanations.csv', index=False)
    assert E.predicted.eq('podium').any() and E.predicted.eq('not podium').any()
    print('\nExplanation fidelity:')
    print(E.groupby('model')[['fidelity_recall_top3', 'fidelity_precision', 'causal_language', 'latency_s']]
          .mean().round(3).to_string())

    # ---------------------------------------------------------------- 4. error analysis by stage
    errs = []
    for p in pipe:
        stage = ''
        if p['expected_kind'] == 'refuse' and p['kind'] != 'refuse':
            stage = 'intent (out of scope not refused)'
        elif p['expected_kind'] != p['kind']:
            stage = 'intent' if p['intent'] != p['expected_intent'] else 'entity' if p['kind'] == 'clarify' else \
                'query' if p['kind'] == 'no_data' else 'intent/entity'
        elif p.get('gold_rows') is not None and not p.get('retrieval_ok'):
            stage = 'entity' if p['templates'] and p['gold_template'] in p['templates'] else 'query'
        if stage:
            errs.append({'id': p['id'], 'stage': stage, 'question': p['question'], 'expected': p['expected_kind'],
                         'got': p['kind'], 'intent': p['intent'], 'entities': p['entities']})
    if len(L) and 'correctness' in L:
        for r in L[L.correctness.notna()].itertuples():
            if not r.faithful:
                errs.append({'id': r.id, 'stage': 'hallucination', 'model': r.model, 'prompting': r.prompting,
                             'question': r.question, 'answer': r.answer,
                             'detail': f'extra numbers: {r.extra_numbers} names: {r.extra_names}'})
            elif r.correctness < 1:
                errs.append({'id': r.id, 'stage': 'model misread the rows', 'model': r.model, 'prompting': r.prompting,
                             'question': r.question, 'answer': r.answer, 'detail': f'missing: {r.missing}'})
    ERR = pd.DataFrame(errs)
    ERR.to_csv(out / 'error_analysis.csv', index=False)
    print('\nErrors by stage:')
    print(ERR.groupby('stage').size().to_string() if len(ERR) else 'none')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
