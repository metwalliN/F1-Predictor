"""Checks every Cypher template against an independent pandas computation on the same CSVs.

usage: python tests/test_templates.py <csv_dir> neo4j            (your Milestone 2 graph, credentials from .env)
       python tests/test_templates.py <csv_dir> <kuzu_db_path>   (offline mirror: tests/build_kuzu_graph.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1rag.graph.client import KuzuClient, Neo4jClient         # noqa: E402
from f1rag.retrieval.templates import T, MECHANICAL_STATUSES  # noqa: E402

csv_dir, db_path = sys.argv[1], sys.argv[2]
rd = lambda n: pd.read_csv(Path(csv_dir) / f'{n}.csv', na_values=['\\N'], keep_default_na=False)
races, results, drivers, cons = rd('races'), rd('results'), rd('drivers'), rd('constructors')
circuits, quali, pits, status = rd('circuits'), rd('qualifying'), rd('pit_stops'), rd('status')
res = (results.merge(races[['raceId', 'year', 'circuitId', 'name']], on='raceId')
       .merge(status, on='statusId').merge(cons[['constructorId', 'name']].rename(columns={'name': 'cname'}),
                                         on='constructorId'))
client = Neo4jClient() if db_path == 'neo4j' else KuzuClient(db_path)
failures = []


def check(tid, params, expected):
    t = T[tid]
    rows = client.run(t.cypher, {**t.fixed_params, **params})
    got = pd.DataFrame(rows)
    for col, val in expected.items():
        g = got[col].tolist() if len(got) else []
        exp = val if isinstance(val, list) else [val]
        ok = len(g) == len(exp) and all(
            (a == b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and np.isclose(a, b))
            or (isinstance(a, list) and sorted(a) == sorted(b)) for a, b in zip(g, exp))
        if not ok:
            failures.append(f'{tid} {params} {col}: graph {g} vs pandas {exp}')
    print(f'{"PASS" if not any(f.startswith(tid + " " + str(params)) for f in failures) else "FAIL"}  {tid} {params}')


def race_id(year, name):
    return int(races[(races.year == year) & (races.name == name)].raceId.iloc[0])


# driver_season_stats
for d, y0, y1 in [(844, 2022, 2022), (1, 2014, 2020), (830, 2015, 2024)]:
    r = res[(res.driverId == d) & res.year.between(y0, y1)]
    check('driver_season_stats', dict(driverId=d, year_from=y0, year_to=y1), dict(
        races_entered=r.raceId.nunique(), wins=int((r.positionOrder == 1).sum()),
        podiums=int((r.positionOrder <= 3).sum()), points=float(r.points.sum()),
        starts_from_pole=int((r.grid == 1).sum()), not_classified=int(r.position.isna().sum()),
        avg_classified_finish=round(float(r.position.mean()), 2)))
# driver_at_circuit
for d, c, y0, y1 in [(1, 9, 1950, 2019), (30, 6, 1950, 2024), (844, 14, 2019, 2024)]:
    r = res[(res.driverId == d) & (res.circuitId == c) & res.year.between(y0, y1)]
    check('driver_at_circuit', dict(driverId=d, circuitId=c, year_from=y0, year_to=y1), dict(
        races_entered=r.raceId.nunique(), wins=int((r.positionOrder == 1).sum()),
        podiums=int((r.positionOrder <= 3).sum()), best_finish=int(r.positionOrder.min())))
# constructor_stats
for k, y0, y1 in [(3, 2014, 2021), (6, 2022, 2024), (131, 2014, 2024)]:
    r = res[(res.constructorId == k) & res.year.between(y0, y1)]
    check('constructor_stats', dict(constructorId=k, year_from=y0, year_to=y1), dict(
        races=r.raceId.nunique(), car_entries=len(r), wins=int((r.positionOrder == 1).sum()),
        points=float(r.points.sum()), classified_finishes=int(r.position.notna().sum()),
        classified_rate=round(r.position.notna().mean(), 3),
        mechanical_retirements=int(r.status.isin(MECHANICAL_STATUSES).sum())))
# head_to_head
for d1, d2, k, y0, y1 in [(846, 817, 1, 1950, 2024), (1, 822, None, 1950, 2024), (830, 815, None, 2016, 2024)]:
    a = res[res.driverId == d1][['raceId', 'constructorId', 'positionOrder', 'year']]
    b = res[res.driverId == d2][['raceId', 'constructorId', 'positionOrder']]
    m = a.merge(b, on=['raceId', 'constructorId'], suffixes=('1', '2'))
    m = m[m.year.between(y0, y1)]
    if k:
        m = m[m.constructorId == k]
    g = m.groupby('constructorId').agg(n=('raceId', 'nunique'), a1=('positionOrder1', lambda s: 0),
                                       ).reset_index()
    g['a1'] = [int((m[m.constructorId == c].positionOrder1 < m[m.constructorId == c].positionOrder2).sum()) for c in g.constructorId]
    g = g.sort_values('n', ascending=False)
    tid, p = ('head_to_head_at_constructor', dict(constructorId=k)) if k else ('head_to_head', {})
    check(tid, dict(driver1Id=d1, driver2Id=d2, year_from=y0, year_to=y1, **p),
          dict(shared_races=g.n.tolist(), driver1_ahead=g.a1.tolist()))
# race_podium / grid gains / qualifying top / pit stops
for y, n in [(2021, 'Italian Grand Prix'), (2019, 'German Grand Prix'), (2008, 'British Grand Prix')]:
    rid = race_id(y, n)
    r = res[res.raceId == rid].sort_values('positionOrder')
    check('race_podium', dict(raceId=rid, top_n=3), dict(finish=[1, 2, 3], grid=r.grid.head(3).tolist()))
    gg = r[(r.grid > 0) & r.position.notna()].assign(g=lambda d: d.grid - d.position)
    gg = gg.sort_values(['g', 'position'], ascending=[False, True]).head(5)
    check('race_grid_gains', dict(raceId=rid), dict(places_gained=gg.g.astype(int).tolist()))
    q = quali[(quali.raceId == rid) & (quali.position <= 3)].sort_values('position')
    check('race_qualifying_top', dict(raceId=rid), dict(qualifying_position=q.position.tolist()))
for y, n in [(2023, 'Qatar Grand Prix'), (2021, 'Italian Grand Prix'), (2008, 'British Grand Prix')]:
    rid = race_id(y, n)
    p = pits[pits.raceId == rid].sort_values('milliseconds').head(5)
    check('race_fastest_pit_stops', dict(raceId=rid), dict(milliseconds=p.milliseconds.tolist()))
    s = pits[pits.raceId == rid].groupby('driverId').size()
    exp = dict(total_stops=int(s.sum()), drivers_who_stopped=len(s)) if len(s) else dict(total_stops=[None])
    check('race_pit_summary', dict(raceId=rid), exp if len(s) else {})
for d, y0, y1 in [(844, 2022, 2022), (1, 2011, 2024)]:
    p = pits.merge(races[['raceId', 'year']])
    p = p[(p.driverId == d) & p.year.between(y0, y1)]
    check('driver_pit_stops', dict(driverId=d, year_from=y0, year_to=y1), dict(
        races_with_stops=p.raceId.nunique(), total_stops=len(p), fastest_stop_ms=int(p.milliseconds.min())))
# qualifying record
for d, y0, y1 in [(844, 2022, 2022), (1, 2007, 2024)]:
    q = quali.merge(races[['raceId', 'year']])
    q = q[(q.driverId == d) & q.year.between(y0, y1)]
    check('driver_qualifying_record', dict(driverId=d, year_from=y0, year_to=y1), dict(
        qualifying_sessions=len(q), poles=int((q.position == 1).sum()), q3_appearances=int(q.q3.notna().sum())))
# circuits
for c in [9, 14, 6]:
    r = races[races.circuitId == c]
    check('circuit_info', dict(circuitId=c), dict(races_held=len(r), first_year=int(r.year.min()),
                                                  last_year=int(r.year.max())))
    w = res[(res.circuitId == c) & (res.positionOrder == 1)].groupby('driverId').size().sort_values(ascending=False)
    check('circuit_top_winners', dict(circuitId=c), dict(wins=w.head(5).tolist()))
# seasons
for y in [2021, 1988, 2024]:
    r = res[res.year == y]
    top = r.groupby('driverId').points.sum().sort_values(ascending=False).head(5)
    check('season_driver_points', dict(year=y), dict(points=top.tolist()))
    topc = r.groupby('constructorId').points.sum().sort_values(ascending=False).head(5)
    check('season_constructor_points', dict(year=y), dict(points=topc.tolist()))
    check('season_overview', dict(year=y), dict(races=r.raceId.nunique(), drivers=r.driverId.nunique(),
                                                constructors=r.constructorId.nunique()))
for y0, y1 in [(2020, 2021), (2023, 2024)]:
    a = res[res.year == y0].groupby('driverId').cname.apply(set)
    b = res[res.year == y1].groupby('driverId').cname.apply(set)
    both = a.index.intersection(b.index)
    changed = [d for d in both if not (a[d] & b[d])]
    check('season_team_changes', dict(year_from=y0, year_to=y1), dict(teams_after=[sorted(b[d]) for d in sorted(
        changed, key=lambda d: (drivers.set_index('driverId').loc[d, 'forename'] + ' ' + drivers.set_index('driverId').loc[d, 'surname']))]))

print()
print('ALL TEMPLATE CHECKS PASSED' if not failures else 'FAILURES:\n' + '\n'.join(failures))
sys.exit(1 if failures else 0)
