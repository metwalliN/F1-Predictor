"""One chat turn through the whole pipeline: preprocessing (entities, intent, session cache, clarification) ->
retrieval (templates) -> language models -> result for the interface, plus the turn log.

Session cache rules (Section 1.b):
- the cache keeps the last identified entities (driver, second driver, constructor, circuit, race, season);
- a follow-up reuses the cache and replaces only the slots the new message names
  ("And in 2020?" replaces the season, "What about Norris?" the driver);
- a required slot the message does not name ("How many points did he score in 2019?") is taken from the cache;
- optional slots (the season) are carried over only on a follow-up or when the message uses a pronoun;
- an out-of-scope message leaves the cache untouched.
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from . import config
from .preprocessing.entities import EntityExtractor, Entity, norm
from .retrieval.planner import plan, required_slots, execute, NO_DATA
from .llm.answer import answer as llm_answer

REFUSAL = ("I don't have the knowledge to answer that. I can answer questions about Formula 1 race results, "
           'drivers, teams, circuits, qualifying, pit stops and seasons from 1950 to 2024, and predict podiums.')
REPHRASE = ("I'm not sure what you're asking. Could you rephrase it? For example: \"How many podiums did Leclerc "
            'get in 2022?" or "Who won the 2019 German Grand Prix?"')
SLOT_QUESTIONS = {
    'driver': 'Which driver do you mean?',
    'driver2': 'Which driver should I compare {driver} with?',
    'constructor': 'Which team (constructor) do you mean?',
    'circuit': 'Which circuit do you mean?',
    'race': 'Which race? Please give the Grand Prix and the year, for example "the 2021 Italian Grand Prix".',
    'season': 'Which season do you mean?',
    'race_or_driver': 'Do you mean a specific race (Grand Prix and year) or a driver over a season?',
}


@dataclass
class SessionState:
    cache: dict = field(default_factory=dict)        # slot -> Entity | tuple | value
    cached_intent: str = None
    pending: dict = None                             # clarification in progress
    history: list = field(default_factory=list)      # [{role, content, meta}]
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def cache_view(self):
        out = {}
        for k, v in self.cache.items():
            if isinstance(v, Entity):
                out[k] = f'{v.name} (id {v.id})'
            elif k == 'season' and v:
                out[k] = f'{v[0]}' if v[0] == v[1] else f'{v[0]}-{v[1]}'
            elif v is not None:
                out[k] = v
        if self.cached_intent:
            out['intent'] = self.cached_intent
        return out


@dataclass
class TurnResult:
    kind: str                     # answer | clarify | refuse | rephrase | no_data | prediction
    message: str = ''
    intent: str = None
    decision: dict = None
    entities: dict = None
    slots: dict = None
    retrieval: object = None
    answers: list = field(default_factory=list)
    prefill: dict = None
    log: dict = None


def _slot_json(slots):
    out = {}
    for k, v in slots.items():
        if isinstance(v, Entity):
            out[k] = v.name
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif v is not None:
            out[k] = v
    return out


class Assistant:
    def __init__(self, client, classifier, extractor=None):
        self.client = client
        self.classifier = classifier
        self.extractor = extractor or EntityExtractor()

    # ------------------------------------------------------------------ public
    def handle(self, message, state, models=(), few_shot=False):
        t0 = time.perf_counter()
        self._models, self._few_shot = tuple(models), few_shot
        state.history.append({'role': 'user', 'content': message})
        if state.pending:
            resolved = self._resolve_pending(message, state)
            if resolved is not None:
                res = resolved
            else:
                state.pending = None
                res = self._turn(message, state, models, few_shot)
        else:
            res = self._turn(message, state, models, few_shot)
        res.log = self._log(message, state, res, time.perf_counter() - t0)
        state.history.append({'role': 'assistant', 'content': res.message, 'meta': res})
        return res

    # ------------------------------------------------------------------ main turn
    def _turn(self, message, state, models, few_shot):
        ex = self.extractor.extract(message)
        d = self.classifier.predict(message, ex, state.cached_intent)
        dec_intent, decision = d.intent, asdict(d)
        if decision['status'] == 'out_of_scope':
            return TurnResult('refuse', REFUSAL, 'out_of_scope', decision, ex.to_dict())
        if decision['status'] == 'rephrase' or dec_intent is None:
            return TurnResult('rephrase', REPHRASE, None, decision, ex.to_dict())

        follow_up = decision['status'] == 'follow_up'
        slots = self._merge(dec_intent, ex, state, follow_up)

        if ex.ambiguities:
            amb = ex.ambiguities[0]
            slot = 'driver2' if amb['type'] == 'driver' and slots.get('driver') and dec_intent == 'head_to_head' \
                else amb['type']
            names = '; '.join(f"{c['name']} ({c['detail']})" for c in amb['candidates'])
            q = f"\"{amb['mention'].title()}\" matches more than one {amb['type']}: {names}. Which one do you mean?"
            state.pending = {'kind': 'ambiguity', 'slot': slot, 'candidates': amb['candidates'], 'intent': dec_intent,
                             'message': message, 'slots': slots, 'mention': amb['mention']}
            return TurnResult('clarify', q, dec_intent, decision, ex.to_dict(), _slot_json(slots))

        self._resolve_race(dec_intent, ex, slots, state, follow_up)
        if slots.get('race_ambiguous'):
            opts = slots.pop('race_ambiguous')
            q = 'That circuit held more than one race that year: ' + '; '.join(o.name for o in opts) + \
                '. Which one do you mean?'
            state.pending = {'kind': 'race_choice', 'slot': 'race', 'options': opts, 'intent': dec_intent,
                             'message': message, 'slots': slots}
            return TurnResult('clarify', q, dec_intent, decision, ex.to_dict(), _slot_json(slots))

        missing = required_slots(dec_intent, slots)
        if missing:
            slot = missing[0]
            q = SLOT_QUESTIONS[slot].format(driver=slots['driver'].name if slots.get('driver') else 'that driver')
            if slot == 'race' and slots.get('race_name_pending'):
                q = f"Which year's {slots['race_name_pending']} do you mean?"
            state.pending = {'kind': 'missing', 'slot': slot, 'intent': dec_intent, 'message': message,
                             'slots': slots}
            return TurnResult('clarify', q, dec_intent, decision, ex.to_dict(), _slot_json(slots))

        # the turn is complete: update the cache (never on out-of-scope turns, which returned above)
        self._update_cache(state, dec_intent, slots)

        if dec_intent == 'podium_prediction':
            prefill = {'driver': slots['driver'], 'race': slots['race'],
                       'user_values': {k: v for k, v in (('grid_position', ex.grid), ('pit_lane_start',
                                       1 if ex.pit_lane_start else None), ('quali_position', ex.quali))
                                       if v is not None}}
            msg = (f"Opening the prediction view for {slots['driver'].name} at the {slots['race'].name}. "
                   'Values you gave are filled in; the rest are defaults shown in gray, and you can edit any of them '
                   'before pressing Predict.')
            return TurnResult('prediction', msg, dec_intent, decision, ex.to_dict(), _slot_json(slots),
                              prefill=prefill)

        calls = plan(dec_intent, slots, norm(message))
        retrieval = execute(self.client, calls)
        if retrieval.error:
            return TurnResult('no_data', 'The knowledge graph could not be reached, so I cannot answer right now.',
                              dec_intent, decision, ex.to_dict(), _slot_json(slots), retrieval)
        if retrieval.empty:
            return TurnResult('no_data', NO_DATA, dec_intent, decision, ex.to_dict(), _slot_json(slots), retrieval)
        ents = _slot_json(slots)
        answers = [llm_answer(m, message, dec_intent, ents, retrieval.calls, few_shot) for m in models]
        return TurnResult('answer', answers[0].text if answers else '', dec_intent, decision, ex.to_dict(), ents,
                          retrieval, answers)

    # ------------------------------------------------------------------ slots and cache
    def _merge(self, intent, ex, state, follow_up):
        cache = state.cache
        new = {}
        if ex.drivers:
            new['driver'] = ex.drivers[0]
            if len(ex.drivers) > 1:
                new['driver2'] = ex.drivers[1]
        if ex.constructors:
            new['constructor'] = ex.constructors[0]
        if ex.circuits:
            new['circuit'] = ex.circuits[0]
        if ex.races:
            new['race'] = ex.races[0]
        if ex.race_names:
            new['race_name_pending'] = ex.race_names[0]
        if ex.season:
            new['season'] = ex.season
        if ex.top_n:
            new['top_n'] = ex.top_n

        if follow_up:
            slots = {k: v for k, v in cache.items()}
            if intent == 'head_to_head' and 'driver' in new and 'driver2' not in new and slots.get('driver2'):
                new['driver2'] = new.pop('driver')              # "What about Norris?" swaps the comparison partner
            if 'season' in new and 'race' not in new and 'race_name_pending' not in new and slots.get('race'):
                # "And in 2020?" for a race question: same Grand Prix, new year
                gp = slots['race'].name.split(' ', 1)[1]
                slots.pop('race')
                if ex.season[0] == ex.season[1]:
                    r = self.extractor.race_from_name(gp, ex.season[0])
                    if r:
                        new['race'] = r
                    else:
                        new['race_name_pending'] = gp
            if 'circuit' in new and slots.get('race') and 'race' not in new:
                slots.pop('race')                               # "And at Monza?" for a race question
            slots.update(new)
        else:
            slots = dict(new)
            carry_optional = ex.pronoun
            for s in self._required(intent):
                if s not in slots and s in cache and s != 'driver2':
                    slots[s] = cache[s]
            if ex.pronoun and 'driver' not in slots and cache.get('driver'):
                slots['driver'] = cache['driver']
            if carry_optional and 'season' not in slots and cache.get('season'):
                slots['season'] = cache['season']
        return slots

    @staticmethod
    def _required(intent):
        return {'driver_stats': ['driver'], 'driver_at_circuit': ['driver', 'circuit'],
                'constructor_stats': ['constructor'], 'head_to_head': ['driver', 'driver2'],
                'race_result': ['race'], 'circuit_info': ['circuit'], 'season_summary': ['season'],
                'qualifying_grid': ['driver'], 'pit_stops': ['driver'],
                'podium_prediction': ['driver', 'race']}.get(intent, [])

    def _resolve_race(self, intent, ex, slots, state, follow_up=False):
        needs_race = intent in ('race_result', 'podium_prediction') or (
            intent in ('qualifying_grid', 'pit_stops') and (slots.get('race_name_pending') or
                                                            (slots.get('circuit') and slots.get('season'))))
        if not needs_race or (slots.get('race') and not slots.get('race_name_pending')):
            return
        season = slots.get('season')
        year = season[0] if season and season[0] == season[1] else None
        if slots.get('race_name_pending') and year:
            r = self.extractor.race_from_name(slots['race_name_pending'], year)
            if r:
                slots['race'] = r
                slots.pop('race_name_pending')
            else:
                slots['race'] = Entity('race', -1, f"{year} {slots.pop('race_name_pending')}", 'no such race')
        elif slots.get('circuit') and year and not slots.get('race_name_pending'):
            options = self.extractor.races_at_circuit(slots['circuit'].id, year)
            if len(options) == 1:
                slots['race'] = options[0]
            elif len(options) > 1:
                slots['race_ambiguous'] = options
            else:
                slots['race'] = Entity('race', -1, f"{year} race at {slots['circuit'].name}", 'no such race')
        if intent in ('qualifying_grid', 'pit_stops') and slots.get('race') and not follow_up and \
                not (ex.races or ex.race_names or ex.circuits):
            slots.pop('race')                                    # a driver-season question, not a race

    def _update_cache(self, state, intent, slots):
        for k in ('driver', 'driver2', 'constructor', 'circuit', 'race', 'season'):
            v = slots.get(k)
            if v is not None and not (isinstance(v, Entity) and v.id < 0):   # never cache a race that does not exist
                state.cache[k] = v
        state.cached_intent = intent

    # ------------------------------------------------------------------ clarification replies
    def _resolve_pending(self, message, state):
        p = state.pending
        text = norm(message)
        choice = None
        if p['kind'] == 'ambiguity':
            for i, c in enumerate(p['candidates'], 1):
                cn = norm(c['name'])
                if text in (str(i), cn) or (text and (text in cn.split() or text == cn.split()[0])):
                    choice = c
                    break
            if choice is None:
                ex = self.extractor.extract(message)
                ids = [e.id for e in ex.drivers + ex.constructors]
                choice = next((c for c in p['candidates'] if c['id'] in ids), None)
            if choice is None:
                return None
            ent = Entity(p['slot'] if p['slot'] != 'driver2' else 'driver', choice['id'], choice['name'],
                         p['mention'], 'chosen by the user')
            if p['slot'] == 'constructor':
                ent.type = 'constructor'
            override = {**{k: v for k, v in p['slots'].items()}, p['slot']: ent}
            state.pending = None
            return self._rerun(p, override, state)
        if p['kind'] == 'race_choice':
            for i, o in enumerate(p['options'], 1):
                if text == str(i) or norm(o.name.split(' ', 1)[1]) in text or text in norm(o.name):
                    choice = o
                    break
            if choice is None:
                return None
            state.pending = None
            return self._rerun(p, {**p['slots'], 'race': choice}, state)
        if p['kind'] == 'missing':
            ex = self.extractor.extract(message)
            slot, slots = p['slot'], dict(p['slots'])
            if slot in ('driver', 'driver2') and ex.drivers:
                slots[slot] = ex.drivers[0]
            elif slot == 'constructor' and ex.constructors:
                slots[slot] = ex.constructors[0]
            elif slot == 'circuit' and ex.circuits:
                slots[slot] = ex.circuits[0]
            elif slot == 'season' and ex.season:
                slots[slot] = ex.season
            elif slot == 'race' and (ex.races or ex.race_names or ex.season or ex.circuits):
                if ex.races:
                    slots['race'] = ex.races[0]
                    slots.pop('race_name_pending', None)
                else:
                    if ex.race_names:
                        slots['race_name_pending'] = ex.race_names[0]
                    if ex.circuits:
                        slots['circuit'] = ex.circuits[0]
                    if ex.season:
                        slots['season'] = ex.season
            elif slot == 'race_or_driver' and (ex.drivers or ex.races or ex.race_names):
                if ex.drivers:
                    slots['driver'] = ex.drivers[0]
                if ex.races:
                    slots['race'] = ex.races[0]
                if ex.season:
                    slots['season'] = ex.season
            elif ex.ambiguities:
                state.pending = None
                return None
            else:
                return None
            state.pending = None
            return self._rerun(p, slots, state)
        return None

    def _rerun(self, p, slots, state):
        """Continue the original question with the slots completed by the clarification."""
        ex_dummy = self.extractor.extract('')
        decision = {'intent': p['intent'], 'status': 'clarified', 'method': 'clarification',
                    'reason': f"user supplied {p['slot']}"}
        intent = p['intent']
        self._resolve_race(intent, self.extractor.extract(p['message']), slots, state)
        if slots.get('race_ambiguous'):
            opts = slots.pop('race_ambiguous')
            state.pending = {'kind': 'race_choice', 'slot': 'race', 'options': opts, 'intent': intent,
                             'message': p['message'], 'slots': slots}
            return TurnResult('clarify', 'Which race: ' + '; '.join(o.name for o in opts) + '?', intent, decision,
                              ex_dummy.to_dict(), _slot_json(slots))
        missing = required_slots(intent, slots)
        if missing:
            slot = missing[0]
            q = SLOT_QUESTIONS[slot].format(driver=slots['driver'].name if slots.get('driver') else 'that driver')
            if slot == 'race' and slots.get('race_name_pending'):
                q = f"Which year's {slots['race_name_pending']} do you mean?"
            state.pending = {'kind': 'missing', 'slot': slot, 'intent': intent, 'message': p['message'],
                             'slots': slots}
            return TurnResult('clarify', q, intent, decision, ex_dummy.to_dict(), _slot_json(slots))
        self._update_cache(state, intent, slots)
        if intent == 'podium_prediction':
            ex = self.extractor.extract(p['message'])
            prefill = {'driver': slots['driver'], 'race': slots['race'],
                       'user_values': {k: v for k, v in (('grid_position', ex.grid), ('quali_position', ex.quali))
                                       if v is not None}}
            return TurnResult('prediction', f"Opening the prediction view for {slots['driver'].name} at the "
                              f"{slots['race'].name}.", intent, decision, ex_dummy.to_dict(), _slot_json(slots),
                              prefill=prefill)
        calls = plan(intent, slots, norm(p['message']))
        retrieval = execute(self.client, calls)
        ents = _slot_json(slots)
        if retrieval.error or retrieval.empty:
            return TurnResult('no_data', NO_DATA, intent, decision, ex_dummy.to_dict(), ents, retrieval)
        answers = [llm_answer(m, p['message'], intent, ents, retrieval.calls, self._few_shot)
                   for m in self._models]
        return TurnResult('answer', answers[0].text if answers else '', intent, decision, ex_dummy.to_dict(), ents,
                          retrieval, answers)

    # ------------------------------------------------------------------ log
    def _log(self, message, state, res, seconds):
        rec = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'session': state.session_id,
               'message': message, 'kind': res.kind, 'intent': res.intent, 'decision': res.decision,
               'entities': res.slots, 'extraction': res.entities, 'cache_after': state.cache_view(),
               'reply': res.message, 'seconds': round(seconds, 3)}
        if res.retrieval:
            rec['templates'] = [c['template'] for c in res.retrieval.calls]
            rec['cypher'] = [c['cypher'] for c in res.retrieval.calls]
            rec['params'] = [c['params'] for c in res.retrieval.calls]
            rec['results'] = [c['rows'] for c in res.retrieval.calls]
        rec['answers'] = [{'model': a.model_id, 'answer': a.text, 'ok': a.ok, 'error': a.error,
                           'latency_s': a.latency_s, 'input_tokens': a.input_tokens, 'output_tokens': a.output_tokens,
                           'prompt': a.prompt} for a in res.answers]
        config.LOG_DIR.mkdir(exist_ok=True)
        with open(config.LOG_DIR / 'turns.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        return rec
