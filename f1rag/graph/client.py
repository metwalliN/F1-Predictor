"""Graph clients. The app uses Neo4jClient (the Milestone 2 graph). KuzuClient is only for offline tests:
it loads the same CSVs into an embedded Kùzu database with the Milestone 2 labels, relationship types,
directions and property names, so the Cypher templates can be checked without a Neo4j server."""
import re
import time

from .. import config


class GraphError(RuntimeError):
    pass


class Neo4jClient:
    backend = 'neo4j'

    def __init__(self, uri=None, user=None, password=None, database=None):
        from neo4j import GraphDatabase
        self.database = database or config.NEO4J_DATABASE
        self._driver = GraphDatabase.driver(uri or config.NEO4J_URI,
                                            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD))

    def run(self, cypher, params=None):
        try:
            records, _, _ = self._driver.execute_query(cypher, params or {}, database_=self.database)
        except Exception as e:                               # connection or Cypher error
            raise GraphError(str(e)) from e
        return [_plain(r.data()) for r in records]

    def close(self):
        self._driver.close()


def _plain(value):
    """Neo4j temporal and numeric types -> plain Python, so rows serialise to JSON for the prompt and the log."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'iso_format'):
        return value.iso_format()
    return value


class KuzuClient:
    backend = 'kuzu'

    def __init__(self, db_path):
        import kuzu
        self._db = kuzu.Database(str(db_path), read_only=True)
        self._conn = kuzu.Connection(self._db)

    def run(self, cypher, params=None):
        params = params or {}
        used = set(re.findall(r'\$(\w+)', cypher))           # Kùzu rejects parameters the query does not use
        try:
            res = self._conn.execute(cypher, {k: v for k, v in params.items() if k in used})
        except Exception as e:
            raise GraphError(str(e)) from e
        df = res.get_as_df()
        rows = df.to_dict('records')
        return [{k: _clean_nan(v) for k, v in r.items()} for r in rows]

    def close(self):
        pass


def _clean_nan(v):
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    if hasattr(v, 'tolist'):
        return v.tolist()
    return v


def get_client():
    """Neo4j by default; F1_GRAPH_BACKEND=kuzu:<path> switches to the offline test graph."""
    backend = config.env('F1_GRAPH_BACKEND', 'neo4j')
    if backend.startswith('kuzu:'):
        return KuzuClient(backend.split(':', 1)[1])
    return Neo4jClient()


def timed_run(client, cypher, params):
    t0 = time.perf_counter()
    rows = client.run(cypher, params)
    return rows, time.perf_counter() - t0
