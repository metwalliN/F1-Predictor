"""Intent classifier comparison on the labelled set: accuracy and confusion matrix for both approaches.

usage: python eval/eval_intents.py            (sentence-transformer from EMBEDDING_MODEL)
       python eval/eval_intents.py --offline  (TF-IDF stand-in encoder, only for testing the script)
A prediction of "rephrase" (the classifier asked the user to reword) counts as wrong.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from f1rag import config                                                     # noqa: E402
from f1rag.preprocessing.entities import EntityExtractor                     # noqa: E402
from f1rag.preprocessing.intents import INTENTS                              # noqa: E402
from f1rag.preprocessing.classifiers import (RuleClassifier, EmbeddingClassifier, TfidfEncoder,  # noqa: E402
                                             SentenceTransformerEncoder, mask)


def build_embedding(extractor, offline, min_score=None, margin=None):
    if offline:
        corpus = [mask(e, extractor.extract(e)) for s in INTENTS.values() for e in s['examples']]
        enc = TfidfEncoder(corpus)
        kw = dict(min_score=0.30, margin=0.02)
    else:
        enc = SentenceTransformerEncoder(config.EMBEDDING_MODEL)
        kw = {}
    if min_score is not None:
        kw['min_score'] = min_score
    if margin is not None:
        kw['margin'] = margin
    return EmbeddingClassifier(enc, extractor, **kw)


def run(clf, data, extractor):
    out = []
    for r in data.itertuples():
        cached = r.cached_intent if isinstance(r.cached_intent, str) and r.cached_intent else None
        d = clf.predict(r.message, extractor.extract(r.message), cached)
        out.append({'id': r.id, 'message': r.message, 'label': r.label, 'predicted': d.intent or 'rephrase',
                    'status': d.status, 'confidence': d.confidence, 'reason': d.reason})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--min-score', type=float)
    ap.add_argument('--margin', type=float)
    ap.add_argument('--set', default='intent_labelled', help='intent_labelled or intent_heldout')
    args = ap.parse_args()
    data = pd.read_csv(ROOT / 'eval' / 'testsets' / f'{args.set}.csv', keep_default_na=False)
    counts = data.label.value_counts()
    assert len(data) >= 40 and counts.min() >= 4, counts
    train_examples = {norm_ for s in INTENTS.values() for norm_ in s['examples']}
    assert not set(data.message) & train_examples, 'test questions must differ from the classifier examples'

    extractor = EntityExtractor()
    labels = list(INTENTS) + ['rephrase']
    res_dir = ROOT / 'eval' / 'results'
    res_dir.mkdir(exist_ok=True, parents=True)
    summary = []
    for clf in (RuleClassifier(), build_embedding(extractor, args.offline, args.min_score, args.margin)):
        pred = run(clf, data, extractor)
        acc = (pred.label == pred.predicted).mean()
        cm = pd.crosstab(pd.Categorical(pred.label, labels), pd.Categorical(pred.predicted, labels),
                         rownames=['true'], colnames=['predicted'], dropna=False)
        tag = clf.name + (' (offline stand-in)' if args.offline and clf.name.startswith('embedding') else '')
        print(f'\n=== {tag}: accuracy {acc:.3f} ({(pred.label == pred.predicted).sum()}/{len(pred)})')
        print(cm.loc[:, cm.sum() > 0].to_string())
        wrong = pred[pred.label != pred.predicted]
        if len(wrong):
            print('errors:')
            print(wrong[['id', 'message', 'label', 'predicted', 'reason']].to_string(index=False))
        slug = clf.name.split()[0].replace('-', '_') + '_' + args.set
        pred.to_csv(res_dir / f'intent_predictions_{slug}.csv', index=False)
        cm.to_csv(res_dir / f'intent_confusion_{slug}.csv')
        per_intent = pred.assign(ok=pred.label == pred.predicted).groupby('label').ok.mean().round(3)
        summary.append({'classifier': tag, 'accuracy': round(acc, 3), 'n': len(pred),
                        'asked_to_rephrase': int((pred.predicted == 'rephrase').sum()),
                        'worst_intent': per_intent.idxmin(), 'worst_intent_acc': per_intent.min()})
        _plot(cm, tag, res_dir / f'intent_confusion_{slug}.png')
    pd.DataFrame(summary).to_csv(res_dir / f'intent_summary_{args.set}.csv', index=False)
    print('\n', pd.DataFrame(summary).to_string(index=False))


def _plot(cm, title, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(cm.values, cmap='Blues')
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(cm.index)), cm.index, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm.values[i, j]:
                ax.text(j, i, cm.values[i, j], ha='center', va='center', fontsize=8,
                        color='white' if cm.values[i, j] > cm.values.max() / 2 else 'black')
    ax.set(xlabel='predicted', ylabel='true', title=f'Intent confusion matrix: {title}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
