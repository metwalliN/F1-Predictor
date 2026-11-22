"""Offline copy of the knowledge graph: loads the CSVs into an embedded Kùzu database that mirrors the Milestone 2
Neo4j schema (same labels, relationship types, directions and property names), so the same Cypher templates run
on it. Used for the offline tests and by start.py when Neo4j is not running. The milestone itself runs on Neo4j.
"""
import shutil
from pathlib import Path

import kuzu
import pandas as pd

NODES = {   # label: (csv, primary key, {graph property: (csv column, type)})
    'Season': ('seasons', 'year', {'year': ('year', 'INT64')}),
    'Circuit': ('circuits', 'circuitId', {
        'circuitId': ('circuitId', 'INT64'), 'circuitRef': ('circuitRef', 'STRING'), 'circuit_name': ('name', 'STRING'),
        'circuit_loc': ('location', 'STRING'), 'circuit_country': ('country', 'STRING'),
        'lat': ('lat', 'DOUBLE'), 'lng': ('lng', 'DOUBLE'), 'alt': ('alt', 'DOUBLE')}),
    'Status': ('status', 'statusId', {'statusId': ('statusId', 'INT64'), 'status': ('status', 'STRING')}),
    'Driver': ('drivers', 'driverId', {
        'driverId': ('driverId', 'INT64'), 'driverRef': ('driverRef', 'STRING'), 'number': ('number', 'INT64'),
        'code': ('code', 'STRING'), 'forename': ('forename', 'STRING'), 'surname': ('surname', 'STRING'),
        'dob': ('dob', 'STRING'), 'nationality': ('nationality', 'STRING')}),
    'Constructor': ('constructors', 'constructorId', {
        'constructorId': ('constructorId', 'INT64'), 'constructorRef': ('constructorRef', 'STRING'),
        'constructor_name': ('name', 'STRING'), 'constructor_nationality': ('nationality', 'STRING')}),
    'Race': ('races', 'raceId', {
        'raceId': ('raceId', 'INT64'), 'year': ('year', 'INT64'), 'round': ('round', 'INT64'), 'name': ('name', 'STRING'),
        'date': ('date', 'STRING'), 'time': ('time', 'STRING')}),
    'Result': ('results', 'resultId', {
        'resultId': ('resultId', 'INT64'), 'number': ('number', 'INT64'), 'grid': ('grid', 'INT64'),
        'position': ('position', 'INT64'), 'positionText': ('positionText', 'STRING'),
        'positionOrder': ('positionOrder', 'INT64'), 'points': ('points', 'DOUBLE'), 'laps': ('laps', 'INT64'),
        'time': ('time', 'STRING'), 'milliseconds': ('milliseconds', 'INT64'), 'fastestLap': ('fastestLap', 'INT64'),
        'rank': ('rank', 'INT64'), 'fastestLapTime': ('fastestLapTime', 'STRING'),
        'fastestLapSpeed': ('fastestLapSpeed', 'DOUBLE')}),
    'Qualifying': ('qualifying', 'qualifyId', {
        'qualifyId': ('qualifyId', 'INT64'), 'number': ('number', 'INT64'), 'position': ('position', 'INT64'),
        'q1': ('q1', 'STRING'), 'q2': ('q2', 'STRING'), 'q3': ('q3', 'STRING')}),
    'PitStop': ('pit_stops', 'eventId', {
        'eventId': ('eventId', 'INT64'), 'stop': ('stop', 'INT64'), 'lap': ('lap', 'INT64'), 'time': ('time', 'STRING'),
        'duration': ('duration', 'STRING'), 'milliseconds': ('milliseconds', 'INT64')}),
}
# (type, from label, to label, from-csv, from key col, to key col)
RELS = [
    ('AT_CIRCUIT', 'Race', 'Circuit', 'races', 'raceId', 'circuitId'),
    ('IN_SEASON', 'Race', 'Season', 'races', 'raceId', 'year'),
    ('FOR_RACE', 'Result', 'Race', 'results', 'resultId', 'raceId'),
    ('BY_DRIVER', 'Result', 'Driver', 'results', 'resultId', 'driverId'),
    ('FOR_CONSTRUCTOR', 'Result', 'Constructor', 'results', 'resultId', 'constructorId'),
    ('HAS_STATUS', 'Result', 'Status', 'results', 'resultId', 'statusId'),
    ('FOR_RACE', 'Qualifying', 'Race', 'qualifying', 'qualifyId', 'raceId'),
    ('BY_DRIVER', 'Qualifying', 'Driver', 'qualifying', 'qualifyId', 'driverId'),
    ('FOR_CONSTRUCTOR', 'Qualifying', 'Constructor', 'qualifying', 'qualifyId', 'constructorId'),
    ('FOR_RACE', 'PitStop', 'Race', 'pit_stops', 'eventId', 'raceId'),
    ('BY_DRIVER', 'PitStop', 'Driver', 'pit_stops', 'eventId', 'driverId'),
]


def load(csv_dir):
    t = {}
    for name in {v[0] for v in NODES.values()}:
        t[name] = pd.read_csv(Path(csv_dir) / f'{name}.csv', na_values=['\\N', ''], keep_default_na=False)
    t['pit_stops'] = t['pit_stops'].reset_index(drop=True)
    t['pit_stops']['eventId'] = t['pit_stops'].index + 1
    return t


def build(csv_dir, db_path):
    db_path = Path(db_path)
    for p in (db_path, db_path.with_name(db_path.name + '.wal')):      # a previous, possibly unfinished build
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    tables = load(csv_dir)
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    tmp = db_path.parent / (db_path.name + '_csv')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for label, (csv, pk, props) in NODES.items():
        cols = ', '.join(f'{p} {typ}' for p, (_, typ) in props.items())
        conn.execute(f'CREATE NODE TABLE {label}({cols}, PRIMARY KEY({pk}))')
        df = pd.DataFrame({p: tables[csv][c] for p, (c, _) in props.items()})
        for p, (_, typ) in props.items():
            if typ == 'INT64':
                df[p] = df[p].astype('Int64')
        path = tmp / f'{label}.csv'
        df.to_csv(path, index=False)
        conn.execute(f'COPY {label} FROM "{path.as_posix()}" (HEADER=true)')   # forward slashes: no escapes
    groups = {}
    for typ, a, b, csv, ka, kb in RELS:
        groups.setdefault(typ, []).append((a, b))
    for typ, pairs in groups.items():
        conn.execute(f'CREATE REL TABLE {typ}(' + ', '.join(f'FROM {a} TO {b}' for a, b in pairs) + ')')
    for typ, a, b, csv, ka, kb in RELS:
        df = tables[csv][[ka, kb]].dropna().astype('int64')
        path = tmp / f'{typ}_{a}_{b}.csv'
        df.to_csv(path, index=False)
        conn.execute(f'COPY {typ} FROM "{path.as_posix()}" (HEADER=true, from="{a}", to="{b}")')
    shutil.rmtree(tmp)
    counts = {label: int(conn.execute(f'MATCH (n:{label}) RETURN count(n) AS c').get_as_df().c[0]) for label in NODES}
    rels = int(conn.execute('MATCH ()-[r]->() RETURN count(r) AS c').get_as_df().c[0])
    conn.close()
    db.close()
    return counts, rels
