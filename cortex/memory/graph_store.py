"""
Graph memory backends.

The interface (`GraphStore`) is deliberately small: add a fact, supersede a
fact, list active facts, get an entity's full temporal history, get a
neighbourhood for retrieval, and export a snapshot for visualization. Two
implementations satisfy it:

- InMemoryGraphStore: networkx-backed, persisted to a local JSON file. Zero
  external services. This is the default so the project runs anywhere.
- Neo4jGraphStore: a real graph database via the official bolt driver, wired
  up in docker-compose.yml. Swap in with GRAPH_BACKEND=neo4j.

Facts are never deleted. A superseded fact keeps its row (status=superseded,
superseded_by=<new fact id>) so the full temporal history of "what did we
believe, and when did we stop believing it" is always queryable — this is
what makes the memory genuinely long-term rather than a cache that silently
overwrites itself.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import networkx as nx

from cortex.models import Fact, normalize_entity


class GraphStore(ABC):
    @abstractmethod
    def add_fact(self, fact: Fact) -> None: ...

    @abstractmethod
    def mark_superseded(self, old_fact_id: str, new_fact_id: str) -> None: ...

    @abstractmethod
    def get_active_facts(self) -> list[Fact]: ...

    @abstractmethod
    def get_history(self, subject: str, relation: str | None = None) -> list[Fact]: ...

    @abstractmethod
    def neighborhood(self, entities: list[str], hops: int = 1) -> list[Fact]: ...

    @abstractmethod
    def snapshot(self) -> dict: ...

    def close(self) -> None:  # pragma: no cover - most backends need nothing
        pass


class InMemoryGraphStore(GraphStore):
    def __init__(self, snapshot_path: str | None = None) -> None:
        self._g = nx.MultiDiGraph()
        self._facts: dict[str, Fact] = {}
        self.snapshot_path = snapshot_path
        if snapshot_path and os.path.exists(snapshot_path):
            self._load(snapshot_path)

    # -- writes ---------------------------------------------------------
    def add_fact(self, fact: Fact) -> None:
        self._facts[fact.id] = fact
        s, o = normalize_entity(fact.subject), normalize_entity(fact.object)
        self._g.add_node(s)
        self._g.add_node(o)
        self._g.add_edge(s, o, key=fact.id, fact_id=fact.id)
        self._persist()

    def mark_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        old = self._facts.get(old_fact_id)
        if old is None:
            return
        old.status = "superseded"
        old.superseded_by = new_fact_id
        self._persist()

    # -- reads ------------------------------------------------------------
    def get_active_facts(self) -> list[Fact]:
        return [f for f in self._facts.values() if f.status == "active"]

    def get_history(self, subject: str, relation: str | None = None) -> list[Fact]:
        subj = normalize_entity(subject)
        facts = [
            f
            for f in self._facts.values()
            if normalize_entity(f.subject) == subj and (relation is None or f.relation == relation)
        ]
        return sorted(facts, key=lambda f: f.created_at)

    def neighborhood(self, entities: list[str], hops: int = 1) -> list[Fact]:
        seen_nodes: set[str] = set()
        frontier = {normalize_entity(e) for e in entities if normalize_entity(e) in self._g}
        for _ in range(max(hops, 1)):
            seen_nodes |= frontier
            nxt: set[str] = set()
            for node in frontier:
                nxt |= set(self._g.successors(node)) | set(self._g.predecessors(node))
            frontier = nxt - seen_nodes
        seen_nodes |= frontier

        fact_ids: set[str] = set()
        for u, v, data in self._g.edges(data=True):
            if u in seen_nodes or v in seen_nodes:
                fact_ids.add(data["fact_id"])
        return [self._facts[fid] for fid in fact_ids if self._facts[fid].status == "active"]

    def snapshot(self) -> dict:
        nodes = [{"id": n, "label": n} for n in self._g.nodes()]
        edges = []
        for _, _, data in self._g.edges(data=True):
            f = self._facts[data["fact_id"]]
            edges.append(
                {
                    "id": f.id,
                    "source": normalize_entity(f.subject),
                    "target": normalize_entity(f.object),
                    "relation": f.relation,
                    "status": f.status,
                    "confidence": f.confidence,
                    "source_title": f.source_title,
                    "source_url": f.source_url,
                    "observed_at": f.observed_at,
                }
            )
        return {"nodes": nodes, "edges": edges, "fact_count": len(self._facts)}

    # -- persistence ---------------------------------------------------
    def _persist(self) -> None:
        if not self.snapshot_path:
            return
        os.makedirs(os.path.dirname(self.snapshot_path) or ".", exist_ok=True)
        with open(self.snapshot_path, "w", encoding="utf-8") as fh:
            json.dump([f.to_dict() for f in self._facts.values()], fh, indent=2)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        for row in rows:
            fact = Fact.from_dict(row)
            self._facts[fact.id] = fact
            s, o = normalize_entity(fact.subject), normalize_entity(fact.object)
            self._g.add_node(s)
            self._g.add_node(o)
            self._g.add_edge(s, o, key=fact.id, fact_id=fact.id)


class Neo4jGraphStore(GraphStore):
    """Real Neo4j backend. Facts become (:Entity)-[:REL {props}]->(:Entity)."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "GRAPH_BACKEND=neo4j requires the `neo4j` package: pip install neo4j"
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def add_fact(self, fact: Fact) -> None:
        s, o = normalize_entity(fact.subject), normalize_entity(fact.object)
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $s})
                MERGE (o:Entity {name: $o})
                CREATE (s)-[r:FACT {
                    id: $id, relation: $relation, confidence: $confidence,
                    status: $status, superseded_by: $superseded_by,
                    source_url: $source_url, source_title: $source_title,
                    observed_at: $observed_at, created_at: $created_at
                }]->(o)
                """,
                s=s,
                o=o,
                id=fact.id,
                relation=fact.relation,
                confidence=fact.confidence,
                status=fact.status,
                superseded_by=fact.superseded_by,
                source_url=fact.source_url,
                source_title=fact.source_title,
                observed_at=fact.observed_at,
                created_at=fact.created_at,
            )

    def mark_superseded(self, old_fact_id: str, new_fact_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH ()-[r:FACT {id: $id}]->() "
                "SET r.status = 'superseded', r.superseded_by = $new_id",
                id=old_fact_id,
                new_id=new_fact_id,
            )

    def get_active_facts(self) -> list[Fact]:
        return self._facts_from_query(
            "MATCH (s:Entity)-[r:FACT {status: 'active'}]->(o:Entity) "
            "RETURN s.name AS s, r AS r, o.name AS o"
        )

    def get_history(self, subject: str, relation: str | None = None) -> list[Fact]:
        query = (
            "MATCH (s:Entity {name: $s})-[r:FACT]->(o:Entity) "
            + ("WHERE r.relation = $relation " if relation else "")
            + "RETURN s.name AS s, r AS r, o.name AS o ORDER BY r.created_at"
        )
        params = {"s": normalize_entity(subject)}
        if relation:
            params["relation"] = relation
        return self._facts_from_query(query, **params)

    def neighborhood(self, entities: list[str], hops: int = 1) -> list[Fact]:
        names = [normalize_entity(e) for e in entities]
        query = (
            "MATCH (e:Entity)-[*1.." + str(max(hops, 1)) + "]-(:Entity) "
            "WHERE e.name IN $names "
            "MATCH (s:Entity)-[r:FACT {status:'active'}]->(o:Entity) "
            "WHERE s.name IN $names OR o.name IN $names "
            "RETURN DISTINCT s.name AS s, r AS r, o.name AS o"
        )
        return self._facts_from_query(query, names=names)

    def snapshot(self) -> dict:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r:FACT]->(o:Entity) "
                "RETURN s.name AS s, r AS r, o.name AS o"
            )
            edges, node_set = [], set()
            for row in result:
                node_set.add(row["s"])
                node_set.add(row["o"])
                r = row["r"]
                edges.append(
                    {
                        "id": r["id"],
                        "source": row["s"],
                        "target": row["o"],
                        "relation": r["relation"],
                        "status": r["status"],
                        "confidence": r["confidence"],
                        "source_title": r["source_title"],
                        "source_url": r["source_url"],
                        "observed_at": r["observed_at"],
                    }
                )
        nodes = [{"id": n, "label": n} for n in node_set]
        return {"nodes": nodes, "edges": edges, "fact_count": len(edges)}

    def close(self) -> None:
        self._driver.close()

    def _facts_from_query(self, query: str, **params) -> list[Fact]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            facts = []
            for row in result:
                r = row["r"]
                facts.append(
                    Fact(
                        subject=row["s"],
                        relation=r["relation"],
                        object=row["o"],
                        confidence=r["confidence"],
                        source_url=r["source_url"],
                        source_title=r["source_title"],
                        observed_at=r["observed_at"],
                        id=r["id"],
                        status=r["status"],
                        superseded_by=r.get("superseded_by"),
                        created_at=r["created_at"],
                    )
                )
        return facts


def get_graph_store(backend: str, *, snapshot_path: str, uri: str, user: str, password: str) -> GraphStore:
    if backend == "neo4j":
        return Neo4jGraphStore(uri, user, password)
    return InMemoryGraphStore(snapshot_path=snapshot_path)
