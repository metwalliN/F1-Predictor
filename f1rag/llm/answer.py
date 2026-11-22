"""Grounded answers with a deterministic fallback: if a model call fails, the user sees "API connection error"
and the answer is a direct reading of the retrieved rows, so the facts are still shown."""
from dataclasses import dataclass, asdict

from .prompts import grounded_messages
from .providers import chat, LLMError

API_ERROR = 'API connection error'


@dataclass
class Answer:
    model_id: str
    text: str
    ok: bool
    error: str = ''
    fallback: str = ''
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    prompt: list = None

    def to_dict(self):
        return asdict(self)


def answer(model_id, question, intent, entities, calls, few_shot=False):
    msgs = grounded_messages(question, intent, entities, calls, few_shot)
    fb = read_rows(calls)
    try:
        r = chat(model_id, msgs)
        return Answer(model_id, r.text, True, '', fb, r.latency_s, r.input_tokens, r.output_tokens, msgs)
    except LLMError as e:
        # the interface shows only API_ERROR; the technical reason is kept for the log
        return Answer(model_id, fb, False, f'{API_ERROR} | {e}', fb, 0.0, 0, 0, msgs)


LABELS = {'avg_classified_finish': 'average classified finish', 'starts_from_pole': 'starts from pole',
          'not_classified': 'not classified', 'races_entered': 'races entered', 'car_entries': 'car entries',
          'classified_rate': 'classified rate', 'mechanical_retirements': 'mechanical retirements',
          'shared_races': 'shared races', 'driver1_ahead': 'driver 1 ahead', 'driver2_ahead': 'driver 2 ahead',
          'places_gained': 'places gained', 'qualifying_position': 'qualifying position',
          'duration_seconds': 'duration (s)', 'avg_stops_per_driver': 'average stops per driver',
          'avg_stops_per_race': 'average stops per race', 'fastest_stop_ms': 'fastest stop (ms)',
          'races_held': 'races held', 'front_row_qualifying': 'front-row qualifying', 'q3_appearances': 'Q3 appearances',
          'avg_qualifying_position': 'average qualifying position', 'teams_before': 'team(s) before',
          'teams_after': 'team(s) after'}


def read_rows(calls):
    """Direct reading of the retrieved results: every value, unchanged, labelled with its column."""
    lines = []
    for c in calls:
        rows = c['rows']
        if not rows:
            continue
        lines.append(f"From the knowledge graph ({c['description'].lower()}):")
        for i, r in enumerate(rows, 1):
            cells = '; '.join(f"{LABELS.get(k, k.replace('_', ' '))}: {_fmt(v)}" for k, v in r.items())
            lines.append(f'{i}. {cells}' if len(rows) > 1 else cells)
        lines.append(f"Scope: {c['scope']}")
    return '\n'.join(lines) if lines else 'The knowledge graph has no data for that question.'


def _fmt(v):
    if isinstance(v, list):
        return ', '.join(map(str, v))
    return 'n/a' if v is None else str(v)
