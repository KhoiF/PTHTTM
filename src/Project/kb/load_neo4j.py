"""Nạp Knowledge Graph vào Neo4j. Mọi node mang nhãn config.KG_LABEL (mặc định DKG) để không đụng dữ liệu khác.

Việc nạp chạy trong kb/build_kg.ipynb. Xem thống kê hoặc sao lưu KG đang có trong Neo4j (chạy từ thư mục src/Project):
    python -m kb.load_neo4j --stats
    python -m kb.load_neo4j --export kb/DB_diabet.cypher
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

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


EXPORT_BATCH = 100


def _cypher_value(value) -> str:
    """Giá trị Python → literal Cypher. Chuỗi viết theo kiểu JSON (Cypher hiểu cùng các ký tự thoát \\n, \\", \\uXXXX)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, 6))          # embedding đã chuẩn hóa: 6 chữ số thập phân là đủ
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cypher_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"`{k}`: {_cypher_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"không xuất được kiểu {type(value).__name__}")


def _unwind(rows: list[dict], body: str) -> list[str]:
    return [f"UNWIND [\n  " + ",\n  ".join(_cypher_value(r) for r in batch) + f"\n] AS row\n{body};\n"
            for batch in _batches(rows, EXPORT_BATCH)]


def export_cypher(driver, path: Path) -> tuple[int, int]:
    """Ghi toàn bộ KG (node mang nhãn KG_LABEL) ra một script Cypher dựng lại được database: ràng buộc, chỉ mục,
    node kèm mọi thuộc tính (cả embedding) và quan hệ. Dùng MERGE nên chạy lại nhiều lần không tạo bản trùng."""
    label, prefix = _label(), index_prefix()
    nodes = run(driver, f"MATCH (n:{label}) RETURN labels(n) AS labels, properties(n) AS props ORDER BY n.id")
    rels = run(driver, f"""MATCH (a:{label})-[r]->(b:{label})
                           RETURN a.id AS s, type(r) AS type, b.id AS e, properties(r) AS props
                           ORDER BY type, s, e, r.key""")
    vector = "OPTIONS {indexConfig: {`vector.dimensions`: %d, `vector.similarity_function`: 'cosine'}}" % config.EMBED_DIM
    out = [
        "// " + "=" * 100,
        f"// DB_diabet.cypher: bản sao lưu Knowledge Graph (nhãn {label}) của đồ án",
        "// Hệ đa agent dự đoán nguy cơ bệnh tiểu đường, tri thức lấy từ Nhà thuốc Long Châu",
        f"// Xuất ngày {date.today():%d/%m/%Y}: {len(nodes)} node, {len(rels)} quan hệ, embedding {config.EMBED_MODEL} "
        f"{config.EMBED_DIM} chiều",
        "//",
        "// Nạp lại (Neo4j 5.x trở lên), chạy từ thư mục src/Project:",
        "//     cypher-shell -u neo4j -p <mật khẩu> -f kb/DB_diabet.cypher",
        f"// Script dùng MERGE nên chạy lại nhiều lần không tạo bản trùng. Muốn nạp sạch từ đầu thì xóa trước:",
        f"//     MATCH (n:{label}) DETACH DELETE n;",
        "// Sinh lại file sau khi dựng lại KG:  python -m kb.load_neo4j --export kb/DB_diabet.cypher",
        "// " + "=" * 100,
        "",
        "// ---- 1. Ràng buộc và chỉ mục " + "-" * 70,
        f"CREATE CONSTRAINT {prefix}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE;",
        f"CREATE FULLTEXT INDEX {prefix}_entity_names IF NOT EXISTS FOR (n:{label}Entity) "
        f"ON EACH [n.name, n.aliases_text, n.name_key];",
        f"CREATE VECTOR INDEX {prefix}_chunk_embedding IF NOT EXISTS FOR (n:{label}Chunk) ON n.embedding {vector};",
        f"CREATE VECTOR INDEX {prefix}_entity_embedding IF NOT EXISTS FOR (n:{label}Entity) ON n.embedding {vector};",
        "",
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in nodes:
        groups[tuple(sorted(l for l in record["labels"] if l != label))].append(dict(record["props"]))
    out.append(f"// ---- 2. Node ({len(nodes)}) " + "-" * 80)
    for extra, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append(f"// {':'.join(extra)}: {len(rows)} node")
        out += _unwind(rows, f"MERGE (n:{label} {{id: row.id}})\nSET n:{':'.join(extra)}, n += row")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for record in rels:
        by_type[record["type"]].append({"s": record["s"], "e": record["e"], "props": dict(record["props"])})
    out.append(f"// ---- 3. Quan hệ ({len(rels)}) " + "-" * 78)
    for rtype, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        keyed = all("key" in r["props"] for r in rows)
        merge = f"MERGE (a)-[r:{rtype} {{key: row.props.key}}]->(b)" if keyed else f"MERGE (a)-[r:{rtype}]->(b)"
        out.append(f"// {rtype}: {len(rows)} quan hệ")
        out += _unwind(rows, f"MATCH (a:{label} {{id: row.s}}), (b:{label} {{id: row.e}})\n{merge}\nSET r += row.props")
    out.append("CALL db.awaitIndexes(300);")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(nodes), len(rels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true", help="in số node theo loại và số quan hệ theo kiểu")
    parser.add_argument("--export", type=Path, metavar="FILE", help="sao lưu KG ra script Cypher dựng lại được database")
    args = parser.parse_args()
    if args.export:
        with get_driver() as driver:
            n_nodes, n_rels = export_cypher(driver, args.export)
        print(f"Đã ghi {args.export}: {n_nodes} node, {n_rels} quan hệ, {args.export.stat().st_size / 1e6:.1f} MB")
        return
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
