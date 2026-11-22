"""Entity extraction (Section 1.b): drivers, constructors, circuits, races (Grand Prix name + year) and seasons or
year ranges, matched against lookup tables built from the knowledge graph (scripts/build_lookups.py).

Matching is deterministic: the message is normalised (lower case, accents removed), then scanned left to right for
the longest alias. Race names are matched first, then circuits, constructors and drivers, and a matched span is
never reused, so "Red Bull Ring" is a circuit and not the Red Bull team.

A name that fits more than one driver ("Schumacher") is reported as ambiguous and the caller asks the user. The only
exception is a name shared with someone who has under a tenth of the other's race entries (Lewis Hamilton has 356,
the 1950s driver Duncan Hamilton 5): that resolves to the dominant driver and the trace lists the alternative.
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict

from .. import config

YEAR_RE = r'(19[5-9]\d|20\d\d)'
DOMINANCE = 10          # resolve a shared name only if one candidate has >= 10x the entries of every other

# Surnames that are ordinary English words: matched only when capitalised in the original message.
COMMON_WORD_SURNAMES = {'hill', 'hunt', 'brown', 'bell', 'stewart', 'walker', 'love', 'fisher', 'hall', 'ward',
                        'lewis', 'moss', 'may', 'cross', 'rose', 'green', 'fry', 'miles', 'barber', 'james', 'page',
                        'parsons', 'sutcliffe', 'force', 'mass', 'grand', 'prix', 'young', 'white', 'black', 'king',
                        'long', 'gay', 'will', 'russo', 'taylor', 'jones', 'johnson', 'wilson', 'palmer', 'crawford',
                        'driver', 'bird', 'rich', 'fox', 'wood', 'small', 'gray', 'grey', 'early', 'green'}
# Surnames never matched alone (a month, or clashes with race/circuit words); a full name still matches.
FULL_NAME_ONLY = {'may', 'grand', 'prix', 'force', 'monaco', 'la', 'de', 'van', 'rb'}

NICKNAMES = {'checo': 'perez', 'kimi': 'raikkonen', 'schumi': 'michael schumacher', 'nando': 'alonso',
             'seb': 'vettel', 'lewis': 'hamilton', 'max verstappen': 'max verstappen', 'danny ric': 'ricciardo'}

CONSTRUCTOR_ALIASES = {
    'mercedes amg': 'mercedes', 'mercedes-amg': 'mercedes', 'merc': 'mercedes', 'amg': 'mercedes',
    'scuderia ferrari': 'ferrari', 'redbull': 'red_bull', 'red bull racing': 'red_bull', 'rbr': 'red_bull',
    'aston': 'aston_martin', 'astonmartin': 'aston_martin', 'haas': 'haas', 'alpine': 'alpine',
    'racing bulls': 'rb', 'visa cash app rb': 'rb', 'vcarb': 'rb', 'alphatauri': 'alphatauri',
    'alpha tauri': 'alphatauri', 'toro rosso': 'toro_rosso', 'tororosso': 'toro_rosso', 'stake': 'sauber',
    'kick sauber': 'sauber', 'alfa': 'alfa', 'alfa romeo': 'alfa', 'racingpoint': 'racing_point',
    'force india': 'force_india', 'forceindia': 'force_india', 'lotus f1': 'lotus_f1', 'mclaren': 'mclaren',
    'williams': 'williams', 'brawn': 'brawn', 'brawn gp': 'brawn', 'renault': 'renault', 'ferrari': 'ferrari'}

CIRCUIT_ALIASES = {
    'barcelona': 'catalunya', 'catalunya': 'catalunya', 'montmelo': 'catalunya', 'spa': 'spa',
    'spa francorchamps': 'spa', 'cota': 'americas', 'austin': 'americas', 'circuit of the americas': 'americas',
    'interlagos': 'interlagos', 'sao paulo': 'interlagos', 'yas marina': 'yas_marina', 'abu dhabi': 'yas_marina',
    'jeddah': 'jeddah', 'zandvoort': 'zandvoort', 'red bull ring': 'red_bull_ring', 'spielberg': 'red_bull_ring',
    'hockenheim': 'hockenheimring', 'hockenheimring': 'hockenheimring', 'nurburgring': 'nurburgring',
    'magny cours': 'magny_cours', 'sepang': 'sepang', 'istanbul park': 'istanbul', 'sochi': 'sochi', 'baku': 'baku',
    'singapore': 'marina_bay', 'marina bay': 'marina_bay', 'shanghai': 'shanghai', 'suzuka': 'suzuka',
    'silverstone': 'silverstone', 'monza': 'monza', 'monaco': 'monaco', 'monte carlo': 'monaco',
    'hungaroring': 'hungaroring', 'budapest': 'hungaroring', 'albert park': 'albert_park',
    'melbourne': 'albert_park', 'bahrain': 'bahrain', 'sakhir': 'bahrain', 'montreal': 'villeneuve',
    'gilles villeneuve': 'villeneuve', 'mexico city': 'rodriguez', 'hermanos rodriguez': 'rodriguez',
    'las vegas': 'vegas', 'miami': 'miami', 'losail': 'losail', 'lusail': 'losail', 'qatar': 'losail',
    'portimao': 'portimao', 'algarve': 'portimao', 'mugello': 'mugello', 'imola': 'imola', 'paul ricard': 'ricard',
    'le castellet': 'ricard', 'estoril': 'estoril', 'kyalami': 'kyalami', 'indianapolis': 'indianapolis',
    'valencia': 'valencia', 'fuji': 'fuji', 'yeongam': 'yeongam', 'buddh': 'buddh', 'jerez': 'jerez',
    'adelaide': 'adelaide', 'zolder': 'zolder', 'watkins glen': 'watkins_glen', 'brands hatch': 'brands_hatch'}

STOP = {'the', 'a', 'an', 'of', 'at', 'in', 'on', 'for', 'and', 'or', 'to', 'from', 'did', 'does', 'do', 'how',
        'what', 'who', 'which', 'when', 'where', 'is', 'was', 'were', 'many', 'much', 'his', 'her', 'their', 'he'}


def norm(text):
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace('&amp;', ' and ').replace("'s ", ' ').replace('’s ', ' ')
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


@dataclass
class Entity:
    type: str                 # driver | constructor | circuit | race
    id: int
    name: str
    mention: str
    note: str = ''            # e.g. resolved by dominance, from cache


@dataclass
class Extraction:
    drivers: list = field(default_factory=list)
    constructors: list = field(default_factory=list)
    circuits: list = field(default_factory=list)
    races: list = field(default_factory=list)
    race_names: list = field(default_factory=list)      # GP names seen without a resolvable year yet
    season: tuple = None                                 # (year_from, year_to)
    season_kind: str = ''                                # single | range | before | after | since | career | pair
    years: list = field(default_factory=list)
    grid: int = None
    quali: int = None
    pit_lane_start: bool = False
    top_n: int = None
    ambiguities: list = field(default_factory=list)      # [{type, mention, candidates:[{id,name,detail}]}]
    pronoun: bool = False

    def any_entity(self):
        return bool(self.drivers or self.constructors or self.circuits or self.races or self.race_names
                    or self.season or self.ambiguities)

    def to_dict(self):
        d = asdict(self)
        d['season'] = list(self.season) if self.season else None
        return d


class EntityExtractor:
    def __init__(self, lookup_dir=None):
        lookup_dir = lookup_dir or config.LOOKUP_DIR
        load = lambda n: json.loads((lookup_dir / f'{n}.json').read_text(encoding='utf-8'))
        self.drivers = {d['id']: d for d in load('drivers')}
        self.constructors = {c['id']: c for c in load('constructors')}
        self.circuits = {c['id']: c for c in load('circuits')}
        self.races = load('races')
        self.race_by_name_year = {(r['name'], r['year']): r for r in self.races}
        self._build_aliases()

    # ------------------------------------------------------------------ alias tables
    def _build_aliases(self):
        self.alias = {}                      # normalised alias -> (type, [ids] or race name)

        def add(alias, typ, value):
            a = norm(alias)
            if not a or a in STOP:
                return
            cur = self.alias.get(a)
            if cur and cur[0] == typ and typ != 'race_name':
                if value not in cur[1]:
                    cur[1].append(value)
            elif not cur or _priority(typ) < _priority(cur[0]):
                self.alias[a] = (typ, [value] if typ != 'race_name' else value)

        for name in {r['name'] for r in self.races}:
            add(name, 'race_name', name)
            add(name.replace('Grand Prix', 'GP'), 'race_name', name)
        add('Brazil Grand Prix', 'race_name', 'Brazilian Grand Prix'); add('Brazil GP', 'race_name', 'Brazilian Grand Prix')
        add('Sao Paulo GP', 'race_name', 'São Paulo Grand Prix'); add('Imola GP', 'race_name', 'Emilia Romagna Grand Prix')
        add('Mexico GP', 'race_name', 'Mexican Grand Prix'); add('Austria GP', 'race_name', 'Austrian Grand Prix')

        ref_to_circuit = {c['ref']: c['id'] for c in self.circuits.values()}
        country_count = {}
        for c in self.circuits.values():
            country_count[norm(c['country'])] = country_count.get(norm(c['country']), 0) + 1
        for c in self.circuits.values():
            add(c['name'], 'circuit', c['id'])
            add(c['ref'].replace('_', ' '), 'circuit', c['id'])
            if norm(c['location']) not in ('barcelona', 'california'):
                add(c['location'], 'circuit', c['id'])
            if country_count[norm(c['country'])] == 1 and norm(c['country']) not in ('united states', 'monaco', 'uae'):
                add(c['country'], 'circuit', c['id'])
        for a, ref in CIRCUIT_ALIASES.items():
            if ref in ref_to_circuit:
                self.alias[norm(a)] = ('circuit', [ref_to_circuit[ref]])

        ref_to_cons = {c['ref']: c['id'] for c in self.constructors.values()}
        for c in self.constructors.values():
            add(c['name'], 'constructor', c['id'])
            add(norm(c['name']).replace(' ', ''), 'constructor', c['id'])
            # a short ref ('hill' for Embassy Hill) would hide a driver surname: refs only for long-running teams
            if norm(c['ref']) == norm(c['name']) or c['entries'] >= 100:
                add(c['ref'].replace('_', ' '), 'constructor', c['id'])
        for a, ref in CONSTRUCTOR_ALIASES.items():
            if ref in ref_to_cons:
                self.alias[norm(a)] = ('constructor', [ref_to_cons[ref]])

        self.surname_index = {}
        for d in self.drivers.values():
            full = f"{d['forename']} {d['surname']}"
            add(full, 'driver', d['id'])
            add(d['ref'].replace('_', ' '), 'driver', d['id'])
            s = norm(d['surname'])
            self.surname_index.setdefault(s, []).append(d['id'])
        for s, ids in self.surname_index.items():
            if s in FULL_NAME_ONLY:
                continue
            cur = self.alias.get(s)
            if cur is None or cur[0] == 'driver':
                self.alias[s] = ('driver', sorted(set(ids) | set(cur[1] if cur else [])))
        for nick, target in NICKNAMES.items():
            if norm(target) in self.alias:
                self.alias[norm(nick)] = self.alias[norm(target)]
        self.codes = {}
        for d in self.drivers.values():
            if d.get('code'):
                self.codes.setdefault(d['code'].upper(), []).append(d['id'])
        self.max_len = max(len(a.split()) for a in self.alias)

    # ------------------------------------------------------------------ extraction
    def extract(self, message):
        ex = Extraction()
        text = norm(message)
        tokens = text.split()
        original_caps = {norm(w) for w in re.findall(r"\b[A-Z][\w'’\-]*", message)}
        self._seasons(text, ex)
        self._prediction_facts(text, ex)
        ex.pronoun = bool(re.search(r'\b(he|his|him|they|their|that driver|that team|same driver)\b', text))

        used = [False] * len(tokens)
        i = 0
        while i < len(tokens):
            hit = None
            for n in range(min(self.max_len, len(tokens) - i), 0, -1):
                if any(used[i:i + n]):
                    continue
                cand = ' '.join(tokens[i:i + n])
                if cand in self.alias:
                    typ = self.alias[cand][0]
                    if typ == 'driver' and n == 1 and cand in COMMON_WORD_SURNAMES and cand not in original_caps:
                        continue
                    if n == 1 and re.fullmatch(r'\d+', cand):
                        continue
                    hit = (n, cand)
                    break
            if hit:
                n, cand = hit
                self._record(ex, cand, self.alias[cand])
                for j in range(i, i + n):
                    used[j] = True
                i += n
            else:
                i += 1
        for code in re.findall(r'\b[A-Z]{3}\b', message):
            if code in self.codes and not any(e.id in self.codes[code] for e in ex.drivers):
                self._driver(ex, code, list(self.codes[code]))
        self._resolve_races(ex)
        return ex

    def _record(self, ex, mention, entry):
        typ, val = entry
        if typ == 'race_name':
            if val not in ex.race_names:
                ex.race_names.append(val)
        elif typ == 'circuit':
            c = self.circuits[val[0]]
            if all(e.id != c['id'] for e in ex.circuits):
                ex.circuits.append(Entity('circuit', c['id'], c['name'], mention))
        elif typ == 'constructor':
            self._pick(ex, mention, val, 'constructor')
        elif typ == 'driver':
            self._driver(ex, mention, val)

    def _pick(self, ex, mention, ids, typ):
        table = self.constructors
        ids = self._narrow_by_season(ids, table, ex)
        chosen, note = self._dominant(ids, table)
        if chosen is None:
            ex.ambiguities.append({'type': typ, 'mention': mention,
                                   'candidates': [self._describe(typ, i) for i in ids]})
            return
        if all(e.id != chosen for e in ex.constructors):
            ex.constructors.append(Entity('constructor', chosen, table[chosen]['name'], mention, note))

    def _driver(self, ex, mention, ids):
        ids = self._narrow_by_season(ids, self.drivers, ex)
        chosen, note = self._dominant(ids, self.drivers)
        if chosen is None:
            if not any(a['mention'] == mention for a in ex.ambiguities):
                ex.ambiguities.append({'type': 'driver', 'mention': mention,
                                       'candidates': [self._describe('driver', i) for i in ids]})
            return
        d = self.drivers[chosen]
        if all(e.id != chosen for e in ex.drivers):
            ex.drivers.append(Entity('driver', chosen, f"{d['forename']} {d['surname']}", mention, note))

    def _narrow_by_season(self, ids, table, ex):
        if len(ids) > 1 and ex.season:
            y0, y1 = ex.season
            active = [i for i in ids if table[i]['first_season'] <= y1 and table[i]['last_season'] >= y0]
            if active:
                return active
        return ids

    def _dominant(self, ids, table):
        if len(ids) == 1:
            return ids[0], ''
        ranked = sorted(ids, key=lambda i: -table[i]['entries'])
        top, rest = ranked[0], ranked[1:]
        if all(table[top]['entries'] >= DOMINANCE * max(1, table[r]['entries']) for r in rest):
            others = ', '.join(self._describe('driver' if table is self.drivers else 'constructor', r)['name']
                               for r in rest)
            return top, f'shared name resolved to the far more frequent entrant (also: {others})'
        return None, ''

    def _describe(self, typ, i):
        if typ == 'driver':
            d = self.drivers[i]
            return {'id': i, 'name': f"{d['forename']} {d['surname']}",
                    'detail': f"{d['nationality']}, raced {d['first_season']}-{d['last_season']}"}
        c = self.constructors[i]
        return {'id': i, 'name': c['name'], 'detail': f"entered {c['first_season']}-{c['last_season']}"}

    # ------------------------------------------------------------------ seasons and pre-race facts
    def _seasons(self, text, ex):
        years = [int(y) for y in re.findall(rf'\b{YEAR_RE}\b', text)]
        ex.years = years
        m = (re.search(rf'\bbetween {YEAR_RE} and {YEAR_RE}\b', text)
             or re.search(rf'\bfrom {YEAR_RE} (?:to|until|till|through) {YEAR_RE}\b', text)
             or re.search(rf'\b{YEAR_RE} (?:to|until|through) {YEAR_RE}\b', text)
             or re.search(rf'\b{YEAR_RE} {YEAR_RE}\b', text))           # "2014-2021" after normalisation
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            ex.season, ex.season_kind = (min(a, b), max(a, b)), 'range'
            return
        if m := re.search(rf'\b(before|prior to) {YEAR_RE}\b', text):
            ex.season, ex.season_kind = (config.FIRST_SEASON, int(m.group(2)) - 1), 'before'
        elif m := re.search(rf'\b(after) {YEAR_RE}\b', text):
            ex.season, ex.season_kind = (int(m.group(2)) + 1, config.LAST_SEASON), 'after'
        elif m := re.search(rf'\b(since) {YEAR_RE}\b', text):
            ex.season, ex.season_kind = (int(m.group(2)), config.LAST_SEASON), 'since'
        elif m := re.search(rf'\b(until|up to|by) {YEAR_RE}\b', text):
            ex.season, ex.season_kind = (config.FIRST_SEASON, int(m.group(2))), 'until'
        elif len(set(years)) == 2:
            a, b = sorted(set(years))
            ex.season, ex.season_kind = (a, b), 'pair'
        elif len(set(years)) == 1:
            ex.season, ex.season_kind = (years[0], years[0]), 'single'
        elif re.search(r'\b(career|all time|ever|in total|overall|history)\b', text):
            ex.season, ex.season_kind = (config.FIRST_SEASON, config.LAST_SEASON), 'career'

    def _prediction_facts(self, text, ex):
        if re.search(r'\b(from|on|takes?|took|starting on|starts on|start on) pole\b|\bpole sitter\b', text):
            ex.grid = 1
        if m := re.search(r'\bp ?(\d{1,2})\b', text):
            ex.grid = int(m.group(1))
        if m := re.search(r'\b(?:start(?:s|ing)?|grid|lines? up)\s+(?:from\s+|in\s+|at\s+|position\s+|slot\s+)?'
                          r'(?:p\s?)?(\d{1,2})(?:st|nd|rd|th)?\b', text):
            ex.grid = int(m.group(1))
        if re.search(r'\bpit ?lane start|from the pit ?lane|starts? from the pits?\b', text):
            ex.pit_lane_start = True
        if m := re.search(r'\bqualif(?:ied|ies|ying)\s+(?:in\s+|at\s+|p\s?)?(\d{1,2})(?:st|nd|rd|th)?\b', text):
            ex.quali = int(m.group(1))
        if m := re.search(r'\btop (3|5|10|three|five|ten)\b', text):
            ex.top_n = {'three': 3, 'five': 5, 'ten': 10}.get(m.group(1), None) or int(m.group(1))
        elif re.search(r'\b(points finishers|full results|whole result|all finishers)\b', text):
            ex.top_n = 10
        elif re.search(r'\b(fourth|4th|fifth|5th)\b', text):
            ex.top_n = 5
        if ex.grid is not None and not 1 <= ex.grid <= 30:
            ex.grid = None

    # ------------------------------------------------------------------ races
    def _resolve_races(self, ex):
        years = sorted(set(ex.years))
        for name in list(ex.race_names):
            if len(years) == 1:
                r = self.race_by_name_year.get((name, years[0]))
                if r:
                    ex.races.append(self._race_entity(r, name))
                    ex.race_names.remove(name)

    def race_from_name(self, name, year):
        r = self.race_by_name_year.get((name, year))
        return self._race_entity(r, name) if r else None

    def races_at_circuit(self, circuit_id, year):
        return [self._race_entity(r, f'circuit {circuit_id} {year}') for r in self.races
                if r['circuitId'] == circuit_id and r['year'] == year]

    def _race_entity(self, r, mention):
        return Entity('race', r['id'], f"{r['year']} {r['name']}", mention)

    def race_info(self, race_id):
        return next((r for r in self.races if r['id'] == race_id), None)


def _priority(typ):
    return {'race_name': 0, 'circuit': 1, 'constructor': 2, 'driver': 3}[typ]
