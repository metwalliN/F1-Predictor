"""Builds the entity lookup tables (drivers, constructors, circuits, races) from the knowledge graph once and saves
them as JSON in data/lookups/. The entity extractor matches user text against these tables and passes the matched
graph ids to the Cypher templates.

usage: python scripts/build_lookups.py            (Neo4j from .env)
       F1_GRAPH_BACKEND=kuzu:<path> python scripts/build_lookups.py   (offline test graph)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1rag import config                       # noqa: E402
from f1rag.graph.client import get_client      # noqa: E402

QUERIES = {
    'drivers': """
MATCH (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)
RETURN d.driverId AS id, d.forename AS forename, d.surname AS surname, d.code AS code, d.driverRef AS ref,
       d.nationality AS nationality, min(r.year) AS first_season, max(r.year) AS last_season, count(res) AS entries
ORDER BY id LIMIT 5000""",
    'constructors': """
MATCH (k:Constructor)<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)
RETURN k.constructorId AS id, k.constructor_name AS name, k.constructorRef AS ref,
       min(r.year) AS first_season, max(r.year) AS last_season, count(res) AS entries
ORDER BY id LIMIT 5000""",
    'circuits': """
MATCH (r:Race)-[:AT_CIRCUIT]->(c:Circuit)
RETURN c.circuitId AS id, c.circuit_name AS name, c.circuitRef AS ref, c.circuit_loc AS location,
       c.circuit_country AS country, count(r) AS races, min(r.year) AS first_year, max(r.year) AS last_year
ORDER BY id LIMIT 5000""",
    'races': """
MATCH (r:Race)-[:AT_CIRCUIT]->(c:Circuit)
RETURN r.raceId AS id, r.year AS year, r.round AS round, r.name AS name, r.date AS date, c.circuitId AS circuitId
ORDER BY id LIMIT 5000""",
}


def main():
    client = get_client()
    config.LOOKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        rows = client.run(q)
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float) and v.is_integer() and k not in ('lat', 'lng'):
                    r[k] = int(v)
                if k == 'date' and v is not None:
                    r[k] = str(v)[:10]
        (config.LOOKUP_DIR / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding='utf-8')
        print(f'{name:<13} {len(rows):>5} rows -> {config.LOOKUP_DIR / (name + ".json")}')
    client.close()


if __name__ == '__main__':
    main()
