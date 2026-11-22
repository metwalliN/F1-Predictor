"""Builds the offline Kùzu copy of the graph (Milestone 2 schema) from the CSVs.

usage: python tests/build_kuzu_graph.py <csv_dir> <db_path>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1rag.graph.offline_graph import build  # noqa: E402

if __name__ == '__main__':
    counts, rels = build(sys.argv[1], sys.argv[2])
    print(counts, 'relationships:', rels)
