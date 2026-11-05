"""Gộp các cách gọi khác nhau của cùng một thực thể và tổng hợp quan hệ trùng.

1. Khóa không dấu + từ điển đồng nghĩa (kb.textutil.name_key).
2. Nếu một khóa vừa là RiskFactor vừa là Disease (vd. "Tăng huyết áp") thì dùng Disease.
3. Hai thực thể cùng loại có cosine bge-m3 ≥ AUTO_MERGE thì gộp; trong [REVIEW, AUTO_MERGE) thì theo quyết định duyệt
   tay trong kb/merge_decisions.csv (merge = 1 gộp, 0 giữ riêng); cặp chưa có quyết định ghi ra merge_candidates.csv.
   Không bao giờ gộp hai tên khác nhau về con số hoặc từ trái nghĩa (tuýp 1/tuýp 2, tăng/hạ...).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from kb.textutil import name_key, plain

AUTO_MERGE, REVIEW = 0.95, 0.92
CONTRAST_TOKENS = {"tang", "ha", "cao", "thap", "man", "cap", "tuyp", "1", "2", "thai", "ky", "tre", "em"}


def _blocked(key_a: str, key_b: str) -> bool:
    a, b = set(key_a.split()), set(key_b.split())
    if set(re.findall(r"\d+", key_a)) != set(re.findall(r"\d+", key_b)):
        return True
    return bool((a ^ b) & CONTRAST_TOKENS)


def _pair_key(etype: str, key_a: str, key_b: str) -> tuple[str, str, str]:
    return (etype, *sorted((key_a, key_b)))


def load_merge_decisions(path: Path) -> dict[tuple[str, str, str], bool]:
    """Quyết định duyệt tay các cặp ứng viên gộp: cột merge = 1 (gộp) hoặc 0 (giữ riêng); để trống là chưa duyệt."""
    decisions = {}
    if not Path(path).exists():
        return decisions
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            merge = str(row.get("merge") or "").strip()
            if merge in {"0", "1"}:
                decisions[_pair_key(row["type"], name_key(row["a"]), name_key(row["b"]))] = merge == "1"
    return decisions


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def build_graph(records: list[dict], llm=None, decisions: dict[tuple[str, str, str], bool] | None = None) -> dict:
    mentions: dict[str, Counter] = defaultdict(Counter)  # id -> các cách viết
    types: dict[str, str] = {}
    sources: dict[str, set] = defaultdict(set)
    disease_keys = {name_key(r[side]) for r in records for side in ("head", "tail")
                    if r[f"{side}_type"] == "Disease"}

    def node_id(name: str, etype: str) -> str:
        key = name_key(name)
        if etype == "RiskFactor" and key in disease_keys:
            etype = "Disease"
        nid = f"{etype}:{key}"
        types[nid] = etype
        return nid

    for r in records:
        r["head_id"], r["tail_id"] = node_id(r["head"], r["head_type"]), node_id(r["tail"], r["tail_type"])
        for side in ("head", "tail"):
            mentions[r[f"{side}_id"]][r[side]] += 1
            sources[r[f"{side}_id"]].add(r["source_url"])

    # Gộp theo embedding trong từng loại
    uf, candidates = _UnionFind(), []
    if llm is not None:
        by_type = defaultdict(list)
        for nid in mentions:
            by_type[types[nid]].append(nid)
        for etype, ids in by_type.items():
            if len(ids) < 2:
                continue
            names = [mentions[i].most_common(1)[0][0] for i in ids]
            vectors = llm.embed(names)
            sims = vectors @ vectors.T
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    keys = ids[i].split(":", 1)[1], ids[j].split(":", 1)[1]
                    if sims[i, j] < REVIEW or _blocked(*keys):
                        continue
                    decision = (decisions or {}).get(_pair_key(etype, *keys))
                    if sims[i, j] >= AUTO_MERGE or decision is True:
                        uf.union(ids[i], ids[j])
                    elif decision is None:
                        candidates.append({"type": etype, "a": names[i], "b": names[j], "cosine": round(float(sims[i, j]), 4)})

    groups = defaultdict(list)
    for nid in mentions:
        groups[uf.find(nid)].append(nid)
    canonical = {}
    entities = {}
    for members in groups.values():
        surface = Counter()
        for m in members:
            surface.update(mentions[m])
        # Tên đại diện: cách viết phổ biến nhất, hòa thì chọn tên ngắn hơn
        name = sorted(surface.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        keeper = max(members, key=lambda m: sum(mentions[m].values()))
        for m in members:
            canonical[m] = keeper
        entities[keeper] = {
            "id": keeper, "type": types[keeper], "name": name, "name_key": keeper.split(":", 1)[1],
            "aliases": sorted(set(surface) - {name}), "mention_count": sum(surface.values()),
            "sources": sorted(set().union(*(sources[m] for m in members))),
        }

    relations = {}
    chunk_mentions = defaultdict(set)
    for r in records:
        h, t = canonical[r["head_id"]], canonical[r["tail_id"]]
        if h == t:
            continue
        props = r.get("props") or {}
        key = json.dumps([h, r["relation"], t, props], ensure_ascii=False, sort_keys=True)
        rel = relations.setdefault(key, {"head": h, "relation": r["relation"], "tail": t, "props": props,
                                         "evidence": [], "source_urls": [], "confidence": 0.0, "tasks": []})
        if r["evidence"] not in rel["evidence"]:
            rel["evidence"].append(r["evidence"])
        if r["source_url"] not in rel["source_urls"]:
            rel["source_urls"].append(r["source_url"])
        if r["task"] not in rel["tasks"]:
            rel["tasks"].append(r["task"])
        rel["confidence"] = max(rel["confidence"], r["confidence"])
        chunk_mentions[r["chunk_id"]].update({h, t})

    return {"entities": list(entities.values()), "relations": list(relations.values()),
            "chunk_mentions": {k: sorted(v) for k, v in chunk_mentions.items()},
            "merge_candidates": candidates}


def find_entities(graph: dict, etype: str, patterns: list[str]) -> list[dict]:
    found = [e for e in graph["entities"] if e["type"] == etype and any(p in e["name_key"] for p in patterns)]
    return sorted(found, key=lambda e: -e["mention_count"])


__all__ = ["build_graph", "find_entities", "load_merge_decisions", "plain"]
