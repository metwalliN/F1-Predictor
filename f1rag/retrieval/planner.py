"""Chooses the template(s) for an intent and builds their parameters from the resolved slots, executes them,
and turns graph-shaped rows into a subgraph (nodes, edges, relationship statements) for the retrieval trace."""
import re
from dataclasses import dataclass, field

from .. import config
from ..graph.client import GraphError
from .templates import T

NO_DATA = 'The knowledge graph has no data for that question.'


@dataclass
class TemplateCall:
    template_id: str
    params: dict
    why: str = ''


@dataclass
class Retrieval:
    calls: list = field(default_factory=list)          # [{template, description, cypher, params, rows, seconds, scope}]
    empty: bool = True
    error: str = ''
    subgraph: dict = None

    @property
    def rows(self):
        return [r for c in self.calls for r in c['rows']]


def season_range(slots, default='career'):
    if slots.get('season'):
        return slots['season']
    return (config.FIRST_SEASON, config.LAST_SEASON) if default == 'career' else None


def plan(intent, slots, text):
    """Deterministic template selection. `text` is the normalised message, used only to pick a variant."""
    y0, y1 = season_range(slots)
    sel = []
    if intent == 'driver_stats':
        sel.append(TemplateCall('driver_season_stats', dict(driverId=slots['driver'].id, year_from=y0, year_to=y1),
                                'driver + season range (career when no season is given)'))
    elif intent == 'driver_at_circuit':
        sel.append(TemplateCall('driver_at_circuit', dict(driverId=slots['driver'].id,
                                                          circuitId=slots['circuit'].id, year_from=y0, year_to=y1),
                                'driver + circuit (+ season range)'))
    elif intent == 'constructor_stats':
        sel.append(TemplateCall('constructor_stats', dict(constructorId=slots['constructor'].id,
                                                          year_from=y0, year_to=y1), 'constructor + season range'))
    elif intent == 'head_to_head':
        p = dict(driver1Id=slots['driver'].id, driver2Id=slots['driver2'].id, year_from=y0, year_to=y1)
        if slots.get('constructor'):
            sel.append(TemplateCall('head_to_head_at_constructor', {**p, 'constructorId': slots['constructor'].id},
                                    'two drivers restricted to the named constructor'))
        else:
            sel.append(TemplateCall('head_to_head', p, 'two drivers, every constructor they shared'))
    elif intent == 'race_result':
        sel.append(TemplateCall('race_podium', dict(raceId=slots['race'].id, top_n=slots.get('top_n') or 3),
                                f"race, top {slots.get('top_n') or 3}"))
    elif intent == 'circuit_info':
        cid = slots['circuit'].id
        sel.append(TemplateCall('circuit_info', dict(circuitId=cid), 'location and races held'))
        sel.append(TemplateCall('circuit_top_winners', dict(circuitId=cid), 'most wins at the circuit'))
    elif intent == 'qualifying_grid':
        if slots.get('race'):
            gains = re.search(r'\b(gain\w*|made up|climb\w*|places|positions|through the field)\b', text)
            pole = re.search(r'\b(pole|qualif\w*|q3|front row|quickest|fastest)\b', text)
            if gains or not pole:
                sel.append(TemplateCall('race_grid_gains', dict(raceId=slots['race'].id), 'race: places gained'))
            if pole or not gains:
                sel.append(TemplateCall('race_qualifying_top', dict(raceId=slots['race'].id),
                                        'race: qualifying order'))
        else:
            sel.append(TemplateCall('driver_qualifying_record', dict(driverId=slots['driver'].id,
                                                                     year_from=y0, year_to=y1),
                                    'driver: qualifying record over the season range'))
    elif intent == 'pit_stops':
        if slots.get('race'):
            fastest = re.search(r'\b(fast\w*|quick\w*|slow\w*|best|service\w*)\b', text)
            count = re.search(r'\b(how many|average|per driver|number|strateg\w*|stops per|twice|times)\b', text)
            if fastest or not count:
                sel.append(TemplateCall('race_fastest_pit_stops', dict(raceId=slots['race'].id),
                                        'race: fastest stops'))
            if count or not fastest:
                sel.append(TemplateCall('race_pit_summary', dict(raceId=slots['race'].id), 'race: stops per driver'))
        else:
            sel.append(TemplateCall('driver_pit_stops', dict(driverId=slots['driver'].id, year_from=y0, year_to=y1),
                                    'driver: stops over the season range'))
    elif intent == 'season_summary':
        a, b = slots['season']
        if re.search(r'\b(changed|switched|swapped|moved|new|different) (to (a )?(new|different) )?teams?\b|'
                     r'\bteam changes?\b|\btransfers?\b', text):
            if a == b:
                a = b - 1
            sel.append(TemplateCall('season_team_changes', dict(year_from=a, year_to=b),
                                    f'drivers who changed team from {a} to {b}'))
        elif re.search(r'\bhow many (different )?(drivers|races|teams|constructors|grands prix)\b|'
                       r'\b(races on the|calendar|on the grid|took part|participat\w*)\b', text):
            sel.append(TemplateCall('season_overview', dict(year=b), 'season size'))
        elif re.search(r'\b(team|teams|constructor\w*)\b', text):
            sel.append(TemplateCall('season_constructor_points', dict(year=b), 'constructors by points'))
        else:
            sel.append(TemplateCall('season_driver_points', dict(year=b), 'drivers by points'))
    return sel


def required_slots(intent, slots):
    """Missing required slots, in the order they should be asked for."""
    need = {'driver_stats': ['driver'], 'driver_at_circuit': ['driver', 'circuit'],
            'constructor_stats': ['constructor'], 'head_to_head': ['driver', 'driver2'],
            'race_result': ['race'], 'circuit_info': ['circuit'], 'season_summary': ['season'],
            'podium_prediction': ['driver', 'race']}.get(intent, [])
    missing = [s for s in need if not slots.get(s)]
    if intent in ('qualifying_grid', 'pit_stops') and not slots.get('race') and not slots.get('driver'):
        missing.append('race_or_driver')
    return missing


def execute(client, calls):
    out = Retrieval()
    for c in calls:
        t = T[c.template_id]
        params = {**t.fixed_params, **c.params}
        try:
            import time
            t0 = time.perf_counter()
            rows = client.run(t.cypher, params)
            secs = time.perf_counter() - t0
        except GraphError as e:
            out.error = str(e)
            rows, secs = [], 0.0
        rows = [_tidy(r) for r in rows]
        out.calls.append({'template': t.id, 'description': t.description, 'why': c.why, 'cypher': t.cypher.strip(),
                          'params': {k: v for k, v in c.params.items()},
                          'fixed_params': list(t.fixed_params), 'rows': rows, 'seconds': round(secs, 3),
                          'scope': t.scope, 'hops': t.hops, 'graph_kind': t.graph.get('kind')})
    out.empty = not any(c['rows'] for c in out.calls)
    out.subgraph = build_subgraph(out.calls)
    return out


def _tidy(row):
    clean = {}
    for k, v in row.items():
        if isinstance(v, float) and v.is_integer() and not k.endswith(('points', 'rate', 'finish', 'position')):
            v = int(v)
        clean[k] = v
    return clean


def build_subgraph(calls):
    """Graph-shaped templates -> nodes, edges and (Label)-[:TYPE]->(Label) statements, using only schema types."""
    nodes, edges = {}, []

    def node(label, key, caption):
        nid = f'{label}:{key}'
        nodes.setdefault(nid, {'id': nid, 'label': label, 'caption': str(caption)})
        return nid

    def edge(a, rel, b):
        e = {'source': a, 'type': rel, 'target': b}
        if e not in edges:
            edges.append(e)

    for c in calls:
        kind = c.get('graph_kind')
        for r in c['rows']:
            if kind == 'race_result':
                race = node('Race', f"{r['year']}-{r['race']}", f"{r['year']} {r['race']}")
                res_caption = f"P{r.get('finish')} (grid {r.get('grid')})"
                res = node('Result', f"{r['year']}-{r['race']}-{r['driver']}", res_caption)
                edge(res, 'FOR_RACE', race)
                edge(res, 'BY_DRIVER', node('Driver', r['driver'], r['driver']))
                if r.get('constructor'):
                    edge(res, 'FOR_CONSTRUCTOR', node('Constructor', r['constructor'], r['constructor']))
                if r.get('status'):
                    edge(res, 'HAS_STATUS', node('Status', r['status'], r['status']))
                if r.get('circuit'):
                    edge(race, 'AT_CIRCUIT', node('Circuit', r['circuit'], r['circuit']))
            elif kind == 'qualifying':
                race = node('Race', f"{r['year']}-{r['race']}", f"{r['year']} {r['race']}")
                q = node('Qualifying', f"{r['year']}-{r['race']}-{r['driver']}", f"Q P{r['qualifying_position']}")
                edge(q, 'FOR_RACE', race)
                edge(q, 'BY_DRIVER', node('Driver', r['driver'], r['driver']))
                edge(q, 'FOR_CONSTRUCTOR', node('Constructor', r['constructor'], r['constructor']))
            elif kind == 'pit_stop':
                race = node('Race', f"{r['year']}-{r['race']}", f"{r['year']} {r['race']}")
                p = node('PitStop', f"{r['driver']}-{r['stop_number']}-{r['lap']}",
                         f"stop {r['stop_number']}, lap {r['lap']}: {r['duration_seconds']} s")
                edge(p, 'FOR_RACE', race)
                edge(p, 'BY_DRIVER', node('Driver', r['driver'], r['driver']))
    if not nodes:
        return None
    labels = {n['id']: n['label'] for n in nodes.values()}
    statements = sorted({f"({labels[e['source']]})-[:{e['type']}]->({labels[e['target']]})" for e in edges})
    return {'nodes': list(nodes.values()), 'edges': edges, 'statements': statements}
