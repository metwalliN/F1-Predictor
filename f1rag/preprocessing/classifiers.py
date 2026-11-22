"""Intent classification (Section 1.a): a rule-based classifier and an embedding-similarity classifier, behind one
decision procedure that handles follow-ups, low confidence and out-of-scope messages the same way for both.

Out of scope is never decided by similarity alone. It needs evidence: a topic the graph has no data for (tyres,
designers, salaries, sprints, lap times, ...) or a message with no Formula 1 word and no recognised entity.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from .intents import INTENTS, OUT_OF_SCOPE_TOPICS, F1_VOCAB, FOLLOW_UP
from .entities import norm, YEAR_RE


@dataclass
class IntentDecision:
    intent: str                # chosen intent, or None when the user must rephrase
    status: str                # classified | follow_up | out_of_scope | rephrase
    method: str
    confidence: float = None
    scores: dict = field(default_factory=dict)
    reason: str = ''


def mask(message, extraction):
    """Replace recognised entities with type words, so similarity depends on the question shape, not the names."""
    text = norm(message)
    for e in extraction.races:
        text = text.replace(norm(e.mention), ' grand prix ')
    for name in extraction.race_names:
        text = text.replace(norm(name), ' grand prix ').replace(norm(name.replace('Grand Prix', 'GP')), ' grand prix ')
    for e in extraction.circuits:
        text = text.replace(norm(e.mention), ' circuit ')
    for e in extraction.constructors:
        text = text.replace(norm(e.mention), ' team ')
    for e in extraction.drivers:
        text = text.replace(norm(e.mention), ' driver ')
    for a in extraction.ambiguities:
        text = text.replace(norm(a['mention']), ' driver ' if a['type'] == 'driver' else ' team ')
    text = re.sub(rf'\b{YEAR_RE}\b', ' year ', text)
    return re.sub(r'\s+', ' ', text).strip()


def oos_evidence(message, extraction):
    text = norm(message)
    if re.search(OUT_OF_SCOPE_TOPICS, text):
        return 'asks about a topic the knowledge graph has no data for'
    if not extraction.any_entity() and not re.search(F1_VOCAB, text):
        return 'no Formula 1 term or entity in the message'
    return ''


def names_new_intent(text, cached_intent):
    """True when the message carries a cue for an intent other than the cached one."""
    if re.search(OUT_OF_SCOPE_TOPICS, text):
        return True
    for intent, spec in INTENTS.items():
        if intent != cached_intent and any(re.search(p, text) for p in spec['patterns']):
            return True
    return False


def decide(best, scores, message, extraction, cached_intent, method, confident, confidence):
    """Shared decision procedure after a classifier has produced its best label (None = no confident label)."""
    text = norm(message)
    cue = bool(re.search(FOLLOW_UP, text))
    short_entity_only = extraction.any_entity() and len(text.split()) <= 6 and best is None
    if cached_intent and (cue or short_entity_only) and not names_new_intent(text, cached_intent):
        return IntentDecision(cached_intent, 'follow_up', method, confidence, scores,
                              'no new intent named: keeps the cached intent')
    evidence = oos_evidence(message, extraction)
    if evidence.startswith('asks about'):               # the graph has no data for this topic, whatever the vote
        return IntentDecision('out_of_scope', 'out_of_scope', method, confidence, scores, evidence)
    if best == 'out_of_scope' or (best is None and evidence):
        if evidence:
            return IntentDecision('out_of_scope', 'out_of_scope', method, confidence, scores, evidence)
        return IntentDecision(None, 'rephrase', method, confidence, scores,
                              'closest to out of scope, but nothing in the message confirms it')
    if best and confident:
        return IntentDecision(best, 'classified', method, confidence, scores)
    return IntentDecision(None, 'rephrase', method, confidence, scores, 'no intent is a clear match')


# ---------------------------------------------------------------------------- rule-based
# Paraphrase lexicons: each is a family of words that ask for the same thing.
LEX = {
    'predict': r'\b(predict\w*|prediction|forecast\w*|odds|chances?|probabilit\w*|likel(y|ihood)|estimate\w*|'
               r'expect\w*|podium model|run the model|use the model|will \w+ (finish|make|get|reach)|'
               r'(could|can|would) \w+ (finish|make|get|reach) (on )?the podium)\b',
    'pit': r'\b(pit|pits|pitted|pitting|pit ?stops?|pit ?lane|stops?|service|strateg\w*|tyre changes?)\b',
    'quali': r'\b(pole|poles|pole position|q1|q2|q3|qualif\w*|front row|grid( slot| position)?|'
             r'(gained|gain|made up|climb\w*|moved up)\b.*\b(places|positions|field|grid)|through the field)\b',
    'h2h': r'\b(vs\.?|v|versus|against|head ?to ?head|teammates?|team ?mates?|compare\w*|comparison|beat|'
           r'outscored|outqualified|came out on top|in front more|ahead more|together)\b',
    'team_change': r'\b((changed|switched|moved|swapped) (to (a )?(new|different) )?teams?|new team|team changes?|'
                   r'transfers?)\b',
    'season': r'\b(most points|points (leader|table)|topped|top of the|led the championship|championship leader|'
              r'champion|title|standings|top (drivers|teams|constructors)|how many (different )?(drivers|races|teams|'
              r'constructors)|races on the|calendar|scored the most|collected the most|who (scored|won) the most)\b',
    'circuit': r'\b(where|which (city|country)|what (city|country)|located|location|hosted|held|calendar|'
               r'since when|first (race|grand prix)|facts?|about|most often|most wins|won most|history of)\b',
    'result': r'\b(who won|won|win|winner|victory|triumph\w*|first|second|third|1st|2nd|3rd|top (three|3|ten|10|five|5)|'
              r'podium|finish\w*|result\w*|classification|came)\b',
    'team_perf': r'\b(reliab\w*|dependab\w*|classified|entries|retire\w*|mechanical|failures?|breakdowns?|dnfs?|'
                 r'finishing rate|points|wins?|won|podiums?|victor\w*|score\w*|collect\w*|perform\w*)\b',
    'driver_perf': r'\b(points|wins?|won|podiums?|poles?|victor\w*|numbers|stats|statistics|record|results?|'
                   r'average|best|career|total|tally|count|score\w*|finish\w*|perform\w*|races?|did \w+ do|'
                   r'how (did|has) \w+ (do|done|perform))\b',
}


class RuleClassifier:
    """Ordered rules: paraphrase lexicons checked in priority order, combined with the entity types present."""
    name = 'rule-based'

    def __init__(self):
        self.lex = {k: re.compile(v) for k, v in LEX.items()}

    def predict(self, message, extraction, cached_intent=None):
        text = norm(message)
        ex = extraction
        L = lambda k: bool(self.lex[k].search(text))
        drivers = len(ex.drivers) + sum(a['type'] == 'driver' for a in ex.ambiguities)
        driverish = drivers or ex.pronoun
        has_race = bool(ex.races or ex.race_names)
        race_like = has_race or bool(ex.circuits and ex.season_kind == 'single')
        rules = [
            ('podium_prediction', L('predict') and (driverish or race_like), 'prediction wording'),
            ('out_of_scope', bool(re.search(OUT_OF_SCOPE_TOPICS, text)), 'topic outside the graph'),
            ('pit_stops', L('pit'), 'pit-stop wording'),
            ('qualifying_grid', L('quali'), 'qualifying or grid wording'),
            ('head_to_head', drivers >= 2 or (drivers and L('h2h') and 'teammate' in text), 'two drivers compared'),
            ('season_summary', L('team_change'), 'team-change wording'),
            ('circuit_info', bool(ex.circuits) and not driverish and not has_race and L('circuit')
             and not (ex.season_kind == 'single' and L('result')), 'circuit-fact wording'),
            ('race_result', race_like and not ex.constructors and L('result'), 'race-result wording with a race'),
            ('driver_at_circuit', bool(driverish and ex.circuits and not has_race), 'a driver and a circuit'),
            ('constructor_stats', bool(ex.constructors) and not drivers and L('team_perf'), 'constructor performance'),
            ('season_summary', bool(ex.season) and not driverish and not ex.constructors and L('season'),
             'season-level wording'),
            ('driver_stats', bool(driverish) and L('driver_perf'), 'driver performance wording'),
        ]
        hit, why = next(((i, w) for i, cond, w in rules if cond), (None, ''))
        decision = decide(hit, {}, message, extraction, cached_intent, self.name, hit is not None,
                          1.0 if hit else 0.0)
        if decision.status == 'classified':
            decision.reason = why
        return decision


# ---------------------------------------------------------------------------- embedding similarity
class SentenceTransformerEncoder:
    """Deployed encoder: a small sentence-transformer, run locally (no API key)."""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


class TfidfEncoder:
    """Offline stand-in used only by the unit tests when the sentence-transformer cannot be downloaded."""
    name = 'tfidf-char-ngrams (test stand-in)'

    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True).fit(corpus)

    def encode(self, texts):
        m = self.v.transform(list(texts)).toarray()
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class EmbeddingClassifier:
    name = 'embedding-similarity'

    def __init__(self, encoder, extractor, min_score=0.45, margin=0.04):
        self.encoder, self.extractor = encoder, extractor
        self.min_score, self.margin = min_score, margin
        self.labels, texts = [], []
        for intent, spec in INTENTS.items():
            for e in spec['examples']:
                self.labels.append(intent)
                texts.append(mask(e, extractor.extract(e)))
        self.example_texts = texts
        self.E = encoder.encode(texts)                # embedded once
        self.labels = np.array(self.labels)

    def scores(self, message, extraction):
        q = self.encoder.encode([mask(message, extraction)])[0]
        sims = self.E @ q
        return {i: float(sims[self.labels == i].max()) for i in INTENTS}

    def predict(self, message, extraction, cached_intent=None):
        s = self.scores(message, extraction)
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        (best, b), (_, second) = ranked[0], ranked[1]
        confident = b >= self.min_score and (b - second) >= self.margin
        d = decide(best if confident or best == 'out_of_scope' else None, s, message, extraction, cached_intent,
                   self.name, confident, round(b, 3))
        if d.status == 'rephrase' and not d.reason.startswith('closest'):
            d.reason = (f'best score {b:.2f} for {best}' +
                        (f' is below the minimum {self.min_score}' if b < self.min_score else
                         f' is within {self.margin} of the second-best intent'))
        return d
