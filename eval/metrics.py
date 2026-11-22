"""Answer metrics computed against the retrieved rows (the gold facts are the rows the correct template returns).

- correctness: share of the gold values the question asks for (answer_fields) that appear in the answer
- faithfulness: the answer adds no number and no driver/team/circuit name that is absent from the context
  (rows + question); an answer with any extra number or name is unfaithful
- refusal: the answer says it cannot answer / the data does not cover it
- explanation fidelity: overlap between the features an explanation names and the top local SHAP contributions
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from f1rag.preprocessing.entities import norm          # noqa: E402

LOOKUP = ROOT / 'data' / 'lookups'
_NAMES = None


def known_names():
    """Surnames, constructor names and circuit names from the graph lookups (for spotting invented names)."""
    global _NAMES
    if _NAMES is None:
        d = json.loads((LOOKUP / 'drivers.json').read_text(encoding='utf-8'))
        c = json.loads((LOOKUP / 'constructors.json').read_text(encoding='utf-8'))
        k = json.loads((LOOKUP / 'circuits.json').read_text(encoding='utf-8'))
        modern = [x for x in d if x['last_season'] >= 1980]
        names = {norm(x['surname']) for x in modern if len(norm(x['surname'])) > 3}
        names |= {norm(x['name']) for x in c if x['last_season'] >= 1980}
        names |= {norm(x['name']) for x in k}
        names -= {'force', 'hill', 'brown', 'bell', 'rose', 'may', 'grand', 'will', 'love', 'hall', 'lewis', 'race'}
        _NAMES = names
    return _NAMES


def number_variants(v):
    out = set()
    if isinstance(v, bool) or v is None:
        return out
    if isinstance(v, (int, float)):
        f = float(v)
        for x in (f, f * 100):
            out |= {f'{x:g}', f'{x:.0f}', f'{x:.1f}', f'{x:.2f}', f'{x:.3f}'}
        if abs(f) < 1:
            out.add(f'{f * 100:.1f}'.rstrip('0').rstrip('.'))
        if f.is_integer():
            out.add(str(int(f)))
    elif isinstance(v, str):
        try:
            return number_variants(float(v))
        except ValueError:
            pass
    return {s.rstrip('0').rstrip('.') if '.' in s else s for s in out} | out


def numbers_in(text):
    return [m.replace(',', '') for m in re.findall(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?', text)]


def _norm_num(s):
    s = s.replace(',', '')
    return s.rstrip('0').rstrip('.') if '.' in s else s


ORDINALS = {'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5', 'sixth': '6', 'seventh': '7',
            'eighth': '8', 'ninth': '9', 'tenth': '10', 'once': '1', 'twice': '2', 'one': '1', 'two': '2',
            'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'zero': '0', 'no': '0', 'none': '0', 'eleven': '11', 'twelve': '12'}


def value_in_answer(v, answer):
    a = norm(answer)
    if isinstance(v, list):
        return all(value_in_answer(x, answer) for x in v)
    if isinstance(v, str) and not re.fullmatch(r'-?\d+(\.\d+)?', v.strip()):
        parts = norm(v).split()
        return norm(v) in a or (parts and parts[-1] in a.split())       # surname is enough for a person
    nums = {_norm_num(n) for n in numbers_in(answer)}
    nums |= {ORDINALS[w] for w in re.findall(r'[a-z]+', answer.lower()) if w in ORDINALS}
    nums |= {n for n in re.findall(r'p(\d+)', answer.lower())}
    nums |= {re.sub(r'(st|nd|rd|th)$', '', w) for w in re.findall(r'\d+(?:st|nd|rd|th)', answer.lower())}
    return bool({_norm_num(x) for x in number_variants(v)} & nums)


def gold_values(rows, fields):
    """answer_fields syntax: 'col' (every row), 'col@0' (row 0), 'col@other=val' (row where other == val)."""
    vals = []
    for f in fields:
        col, _, sel = f.partition('@')
        if not sel:
            vals += [(f, r.get(col)) for r in rows]
        elif sel.isdigit():
            if int(sel) < len(rows):
                vals.append((f, rows[int(sel)].get(col)))
        else:
            k, _, want = sel.partition('=')
            vals += [(f, r.get(col)) for r in rows if str(r.get(k)) == want]
    return vals


def correctness(answer, rows, fields):
    vals = gold_values(rows, fields)
    if not vals:
        return None, []
    missing = [(f, v) for f, v in vals if not value_in_answer(v, answer)]
    return 1 - len(missing) / len(vals), missing


def faithfulness(answer, rows, question, entities=None, extra_context=''):
    """extra_context: other text the model was given (the scope notes), whose numbers it may quote."""
    ctx_text = json.dumps(rows, ensure_ascii=False, default=str) + ' ' + question + ' ' + \
        json.dumps(entities or {}, ensure_ascii=False, default=str) + ' ' + extra_context
    allowed = set()
    for r in rows:
        for v in r.values():
            for x in (v if isinstance(v, list) else [v]):
                allowed |= {_norm_num(s) for s in number_variants(x)}
    allowed |= {_norm_num(n) for n in numbers_in(ctx_text)}
    allowed |= {str(i) for i in range(0, max(4, len(rows) + 1))}   # list numbering, ordinals 1st-3rd
    for r in rows:                                             # differences and sums the answer may state
        nums = [float(v) for v in r.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    allowed.add(_norm_num(f'{nums[i] - nums[j]:g}'))
    extra_nums = [n for n in numbers_in(answer) if _norm_num(n) not in allowed]
    ctx = norm(ctx_text)
    a = norm(answer)
    extra_names = [n for n in known_names() if re.search(rf'\b{re.escape(n)}\b', a) and n not in ctx]
    return (not extra_nums and not extra_names), extra_nums, extra_names


REFUSAL_RE = (r"(do(es)? ?n[o']t (have|contain|include|say|cover|show)|not (in|part of) (the )?(data|context|results)|"
              r"no (data|information)|cannot (answer|tell|say|determine)|can't (answer|tell|say|determine)|"
              r"unable to|isn't (in|available)|not available|insufficient|does not (mention|specify|provide))")


def is_refusal(answer):
    return bool(re.search(REFUSAL_RE, answer.lower()))


CAUSAL_RE = (r'\b(because (of )?(his|her|the|a|its|he|she|they)|caused|(?<!not )causes|(?<!not )cause of|led to|'
             r'resulted in|due to|thanks to|made him|guarantee\w*)\b')


def causal_language(text):
    """Flags wording that presents a feature or fact as the cause of a result (a negated "not a cause" is fine)."""
    return bool(re.search(CAUSAL_RE, text.lower()))


def explanation_fidelity(text, contributions, plain, listed, k=3):
    """recall_top: share of the top-k features by |SHAP| that the explanation names;
    precision: share of the features it names that were in the contribution lists given to the model."""
    t = norm(text)
    mentioned = {f for f, label in plain.items() if norm(label) in t}
    keyw = {'grid_position': ['grid', 'starting position', 'starts from', 'start from', 'starting slot'],
            'q1_gap_pct': ['q1 gap', 'qualifying pace', 'gap to the fastest'],
            'driver_podium_rate_last5': ['podium rate, last 5', "driver's podium rate", 'recent podiums'],
            'driver_avg_finish_rel_last5': ['average relative finish', 'recent finishes'],
            'driver_finish_rate_last5': ["driver's finishing rate"],
            'constructor_podium_rate_last5': ["team's podium rate", 'team podium rate'],
            'constructor_best_finish_rel_last5': ["team's best relative finish", 'best relative finish'],
            'constructor_finish_rate_last5': ["team's finishing rate"],
            'constructor_points_share': ["team's points"], 'driver_points_share': ["driver's points"],
            'driver_champ_rank': ["driver's championship position"],
            'constructor_champ_rank': ["team's championship position"]}
    for f, words in keyw.items():
        if any(norm(w) in t for w in words):
            mentioned.add(f)
    top = [f for f, _ in sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:k]]
    hit = len(set(top) & mentioned)
    precision = len(mentioned & set(listed)) / len(mentioned) if mentioned else 0.0
    return {'recall_top': hit / k, 'precision': precision, 'mentioned': sorted(mentioned), 'top': top}
