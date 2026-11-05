"""Nạp Knowledge Graph vào Neo4j. Mọi node mang nhãn config.KG_LABEL (mặc định DKG) để không đụng dữ liệu khác.

Việc nạp chạy trong kb/build_kg.ipynb. Xem thống kê KG đang có trong Neo4j (chạy từ thư mục src/Project):
    python -m kb.load_neo4j --stats
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase

import config
from kb.schema import ENTITY_TYPES, RELATIONS, STRUCTURAL_RELATIONS

NODE_TYPES = set(ENTITY_TYPES) | {"Feature"}
REL_TYPES = set(RELATIONS) | STRUCTURAL_RELATIONS


def _label() -> str:
    label = config.KG_LABEL
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
        raise ValueError(f"KG_LABEL không hợp lệ: {label}")
    return label


def hidden_labels() -> list[str]:
    """Nhãn kỹ thuật của KG (không phải loại thực thể): nhãn KG, Entity, và nhãn riêng để đánh chỉ mục."""
    label = _label()
    return [label, "Entity", f"{label}Entity", f"{label}Chunk"]


def index_prefix() -> str:
    # Chỉ mục đặt trên nhãn riêng của từng KG (DKGEntity, DKGChunk...) để KG thật và KG test (DKGTEST) không lẫn
    # kết quả tìm kiếm: queryNodes lấy top-n trên toàn chỉ mục rồi mới lọc theo nhãn.
    return _label().lower()


def get_driver():
    # Tắt thông báo deprecation/thuộc tính chưa tồn tại của server (không ảnh hưởng kết quả)
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                                notifications_min_severity="OFF")


def run(driver, query: str, **params):
    return driver.execute_query(query, params, database_=config.NEO4J_DATABASE).records


def reset(driver) -> int:
    """Xóa toàn bộ node của đồ án (chỉ những node mang nhãn KG_LABEL)."""
    return run(driver, f"MATCH (n:{_label()}) DETACH DELETE n RETURN count(n) AS n")[0]["n"]


def ensure_schema(driver) -> None:
    label = _label()
    run(driver, f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
    prefix = index_prefix()
    for legacy in ("entity_names", "chunk_embedding", "entity_embedding"):   # chỉ mục dùng chung của bản cũ
        run(driver, f"DROP INDEX {legacy} IF EXISTS")
    run(driver, f"CREATE FULLTEXT INDEX {prefix}_entity_names IF NOT EXISTS FOR (n:{label}Entity) "
                "ON EACH [n.name, n.aliases_text, n.name_key]")
    for name, node in ((f"{prefix}_chunk_embedding", f"{label}Chunk"), (f"{prefix}_entity_embedding", f"{label}Entity")):
        run(driver, f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{node}) ON n.embedding OPTIONS "
                    f"{{indexConfig: {{`vector.dimensions`: {config.EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}")
    run(driver, "CALL db.awaitIndexes(300)")


def _batches(rows: list, size: int = 200):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def load_graph(driver, graph: dict, chunks: list[dict], pages: list[dict], llm) -> None:
    label = _label()
    entities = graph["entities"]
    vectors = llm.embed([f"{e['name']} ({e['type']})" for e in entities]) if entities else []
    by_type = defaultdict(list)
    for entity, vector in zip(entities, vectors):
        if entity["type"] not in NODE_TYPES:
            raise ValueError(f"loại node lạ: {entity['type']}")
        props = {k: v for k, v in entity.items() if k not in {"id", "type"} and v is not None}
        props["aliases_text"] = " | ".join(entity.get("aliases", []))
        props["embedding"] = vector.tolist()
        by_type[entity["type"]].append({"id": entity["id"], "props": props})
    for etype, rows in by_type.items():
        for batch in _batches(rows):
            run(driver, f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n:Entity:{label}Entity:{etype}, n += row.props",
                rows=batch)

    sources = [{"id": p["url"], "props": {"url": p["url"], "title": p["title"], "page_type": p["type"],
                                          "topic": p["topic"]}} for p in pages]
    for batch in _batches(sources):
        run(driver, f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n:Source, n += row.props", rows=batch)

    chunk_vectors = llm.embed([f"{c['heading']}\n{c['text']}" for c in chunks]) if chunks else []
    chunk_rows = [{"id": c["id"], "url": c["url"],
                   "props": {"heading": c["heading"], "text": c["text"], "title": c["title"], "url": c["url"],
                             "embedding": v.tolist()}} for c, v in zip(chunks, chunk_vectors)]
    for batch in _batches(chunk_rows, 50):
        run(driver, f"""UNWIND $rows AS row
            MERGE (c:{label} {{id: row.id}}) SET c:Chunk:{label}Chunk, c += row.props
            WITH c, row MATCH (s:{label}:Source {{id: row.url}}) MERGE (c)-[:FROM]->(s)""", rows=batch)

    mention_rows = [{"chunk": cid, "entity": eid} for cid, eids in graph["chunk_mentions"].items() for eid in eids]
    for batch in _batches(mention_rows, 500):
        run(driver, f"""UNWIND $rows AS row
            MATCH (c:{label}:Chunk {{id: row.chunk}}), (e:{label} {{id: row.entity}}) MERGE (c)-[:MENTIONS]->(e)""",
            rows=batch)

    rel_by_type = defaultdict(list)
    for rel in graph["relations"]:
        if rel["relation"] not in REL_TYPES:
            raise ValueError(f"quan hệ lạ: {rel['relation']}")
        props = {"evidence": rel["evidence"], "source_urls": rel["source_urls"], "confidence": rel["confidence"],
                 "tasks": rel["tasks"], **{k: v for k, v in rel["props"].items() if v is not None}}
        props["key"] = f"{rel['props'].get('op', '')}{rel['props'].get('value', '')}{rel['props'].get('value_max', '')}{rel['props'].get('unit', '')}"
        rel_by_type[rel["relation"]].append({"h": rel["head"], "t": rel["tail"], "props": props})
    for rtype, rows in rel_by_type.items():
        for batch in _batches(rows, 500):
            run(driver, f"""UNWIND $rows AS row
                MATCH (h:{label} {{id: row.h}}), (t:{label} {{id: row.t}})
                MERGE (h)-[r:{rtype} {{key: row.props.key}}]->(t) SET r += row.props""", rows=batch)


def stats(driver) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = _label()
    nodes = run(driver, f"""MATCH (n:{label}) UNWIND [l IN labels(n) WHERE NOT l IN $hidden] AS type
                            RETURN type, count(*) AS count ORDER BY count DESC""", hidden=hidden_labels())
    rels = run(driver, f"MATCH (:{label})-[r]->(:{label}) RETURN type(r) AS relation, count(*) AS count ORDER BY count DESC")
    return pd.DataFrame([dict(r) for r in nodes]), pd.DataFrame([dict(r) for r in rels])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true", help="in số node theo loại và số quan hệ theo kiểu")
    args = parser.parse_args()
    if not args.stats:
        parser.print_help()
        return
    with get_driver() as driver:
        nodes, rels = stats(driver)
    print(f"Nhãn {_label()}: {int(nodes['count'].sum()) if len(nodes) else 0} node, "
          f"{int(rels['count'].sum()) if len(rels) else 0} quan hệ\n")
    print(nodes.to_string(index=False) if len(nodes) else "(chưa có node)")
    print()
    print(rels.to_string(index=False) if len(rels) else "(chưa có quan hệ)")


if __name__ == "__main__":
    main()
