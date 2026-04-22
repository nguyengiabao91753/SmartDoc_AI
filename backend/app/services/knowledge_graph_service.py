from __future__ import annotations

from typing import Any, Dict, List

from neo4j import GraphDatabase

from app.core.config import settings
from app.core.logger import LOG
from app.services.embedding_service import EmbeddingService


class KnowledgeGraphService:
    """Service quan ly Knowledge Graph tren Neo4j, gom ca vector indexes."""

    def __init__(self, embedding_service=None):
        url = settings.NEO4J_URI
        user = settings.NEO4J_USERNAME
        password = settings.NEO4J_PASSWORD

        self.driver = GraphDatabase.driver(
            url,
            auth=(user, password),
            connection_timeout=3.0,
            max_connection_lifetime=60.0,
        )
        self.embedding_service = embedding_service or EmbeddingService()
        self.dim = self.embedding_service.get_dimension()
        self.is_online = True

        import threading

        threading.Thread(target=self._setup_vector_index, daemon=True).start()

    def close(self):
        self.driver.close()

    def _setup_vector_index(self):
        """Create indexes if missing and migrate old records to list-based doc ids."""
        LOG.info("[KnowledgeGraphService] Setting up Neo4j indexes...")
        try:
            with self.driver.session() as session:
                result = session.run("SHOW INDEXES")
                indexes = [record["name"] for record in result]

                if "entity_vector_index" not in indexes:
                    session.run(
                        f"""
                        CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                        FOR (n:Entity) ON (n.embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: {self.dim},
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}
                        """
                    )
                    LOG.info("[KnowledgeGraphService] Created entity_vector_index")

                if "entity_doc_id_index" not in indexes:
                    session.run("CREATE INDEX entity_doc_id_index IF NOT EXISTS FOR (n:Entity) ON (n.document_id)")
                    LOG.info("[KnowledgeGraphService] Created entity_doc_id_index")

                if "community_vector_index" not in indexes:
                    session.run(
                        f"""
                        CREATE VECTOR INDEX community_vector_index IF NOT EXISTS
                        FOR (n:Community) ON (n.embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: {self.dim},
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}
                        """
                    )
                    LOG.info("[KnowledgeGraphService] Created community_vector_index")

                if "community_doc_id_index" not in indexes:
                    session.run("CREATE INDEX community_doc_id_index IF NOT EXISTS FOR (n:Community) ON (n.document_id)")
                    LOG.info("[KnowledgeGraphService] Created community_doc_id_index")

                # Backward compatibility for old records that only had scalar document_id.
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.document_ids IS NULL AND e.document_id IS NOT NULL
                    SET e.document_ids = [toString(e.document_id)]
                    """
                )
                session.run(
                    """
                    MATCH ()-[r:RELATED]-()
                    WHERE r.document_ids IS NULL AND r.document_id IS NOT NULL
                    SET r.document_ids = [toString(r.document_id)]
                    """
                )
                session.run(
                    """
                    MATCH (c:Community)
                    WHERE c.document_ids IS NULL AND c.document_id IS NOT NULL
                    SET c.document_ids = [toString(c.document_id)]
                    """
                )

                result = session.run("SHOW INDEXES")
                updated_indexes = [record["name"] for record in result]
                LOG.info("[KnowledgeGraphService] Active indexes: %s", updated_indexes)
        except Exception as exc:
            LOG.warning("[KnowledgeGraphService] Index setup failed, fallback retrieval will be used: %s", exc)

    def check_if_graph_exists(self) -> bool:
        if not self.is_online:
            return False
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN count(n) > 0 AS exists")
                record = result.single()
                return record["exists"] if record else False
        except Exception as exc:
            LOG.error("[KnowledgeGraphService] Failed to check graph existence: %s", exc)
            return False

    def get_indexed_document_ids(self) -> List[str]:
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WITH coalesce(
                        n.document_ids,
                        CASE
                            WHEN n.document_id IS NULL THEN []
                            ELSE [toString(n.document_id)]
                        END
                    ) AS doc_ids
                    UNWIND doc_ids AS doc_id
                    RETURN DISTINCT toString(doc_id) AS doc_id
                    """
                )
                return [record["doc_id"] for record in result if record["doc_id"]]
        except Exception as exc:
            LOG.error("[KnowledgeGraphService] Failed to list indexed document_ids: %s", exc)
            return []

    def is_document_indexed(self, document_id: str) -> bool:
        if not self.is_online:
            return False
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WHERE n.document_id = $doc_id OR $doc_id IN coalesce(n.document_ids, [])
                    RETURN count(n) > 0 AS exists
                    """,
                    doc_id=str(document_id),
                )
                record = result.single()
                return record["exists"] if record else False
        except Exception as exc:
            LOG.error("[KnowledgeGraphService] Failed to check document_id in graph: %s", exc)
            return False

    def upsert_graph(self, document_id: str, graph_data: Dict[str, Any]):
        """Upsert entities and relationships with batch + UNWIND."""
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])

        if not entities and not relationships:
            return

        from concurrent.futures import ThreadPoolExecutor

        def prepare_entity(ent: Dict[str, Any]) -> Dict[str, Any]:
            original_name = str(ent.get("name", "")).strip()
            if not original_name:
                return {}
            canonical_name = original_name.lower()
            desc = str(ent.get("description", "")).strip()
            label = str(ent.get("type", "Unknown")).strip() or "Unknown"
            emb = self.embedding_service.embed_query(f"{original_name}: {desc}")
            vector = [float(x) for x in emb]
            return {
                "name": canonical_name,
                "original_name": original_name,
                "label": label,
                "description": desc,
                "embedding": vector,
                "document_id": str(document_id),
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            entity_payloads = [payload for payload in executor.map(prepare_entity, entities) if payload]

        with self.driver.session() as session:
            if entity_payloads:
                session.run(
                    """
                    UNWIND $batch AS data
                    MERGE (e:Entity {name: data.name})
                    ON CREATE SET e.original_name = data.original_name
                    SET e.type = data.label,
                        e.description = data.description,
                        e.embedding = data.embedding,
                        e.document_id = coalesce(e.document_id, data.document_id),
                        e.document_ids = CASE
                            WHEN e.document_ids IS NULL THEN [data.document_id]
                            WHEN data.document_id IN e.document_ids THEN e.document_ids
                            ELSE e.document_ids + data.document_id
                        END,
                        e.original_name = coalesce(e.original_name, data.original_name)
                    """,
                    batch=entity_payloads,
                )

            rel_payloads: List[Dict[str, Any]] = []
            for rel in relationships:
                source = str(rel.get("source", "")).strip().lower()
                target = str(rel.get("target", "")).strip().lower()
                if not source or not target:
                    continue
                rel_payloads.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": str(rel.get("relation", "RELATED")).strip() or "RELATED",
                        "description": str(rel.get("description", "")).strip(),
                        "document_id": str(document_id),
                    }
                )

            if rel_payloads:
                session.run(
                    """
                    UNWIND $batch AS data
                    MATCH (a:Entity {name: data.source})
                    MATCH (b:Entity {name: data.target})
                    MERGE (a)-[r:RELATED {type: data.relation}]->(b)
                    SET r.description = data.description,
                        r.document_id = coalesce(r.document_id, data.document_id),
                        r.document_ids = CASE
                            WHEN r.document_ids IS NULL THEN [data.document_id]
                            WHEN data.document_id IN r.document_ids THEN r.document_ids
                            ELSE r.document_ids + data.document_id
                        END
                    """,
                    batch=rel_payloads,
                )

        LOG.info(
            "[KnowledgeGraphService] Upserted %d entities and %d relationships for document %s",
            len(entity_payloads),
            len(relationships),
            document_id,
        )

    @staticmethod
    def _create_entity_node(tx, name, label, description, embedding, doc_id):
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $label,
            e.description = $description,
            e.embedding = $embedding,
            e.document_id = coalesce(e.document_id, $doc_id),
            e.document_ids = CASE
                WHEN e.document_ids IS NULL THEN [$doc_id]
                WHEN $doc_id IN e.document_ids THEN e.document_ids
                ELSE e.document_ids + $doc_id
            END
        """
        tx.run(query, name=name, label=label, description=description, embedding=embedding, doc_id=str(doc_id))

    @staticmethod
    def _create_relationship(tx, source, target, relation, description, doc_id):
        query = """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        MERGE (a)-[r:RELATED {type: $relation}]->(b)
        SET r.description = $description,
            r.document_id = coalesce(r.document_id, $doc_id),
            r.document_ids = CASE
                WHEN r.document_ids IS NULL THEN [$doc_id]
                WHEN $doc_id IN r.document_ids THEN r.document_ids
                ELSE r.document_ids + $doc_id
            END
        """
        tx.run(
            query,
            source=str(source).strip().lower(),
            target=str(target).strip().lower(),
            relation=relation,
            description=description,
            doc_id=str(doc_id),
        )

    def search_local_context(self, query: str, top_k: int = 5) -> str:
        emb = self.embedding_service.embed_query(query)
        query_vector = [float(x) for x in emb]

        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('entity_vector_index', $top_k, $query_vector)
                YIELD node AS startNode, score
                MATCH (startNode)-[r]->(neighbor)
                RETURN startNode.name AS source,
                       coalesce(r.type, type(r)) AS relation,
                       neighbor.name AS target,
                       r.description AS details,
                       score
                ORDER BY score DESC
                """,
                query_vector=query_vector,
                top_k=top_k,
            )

            context_parts = []
            for record in result:
                context_parts.append(
                    f"- {record['source']} co quan he {record['relation']} voi {record['target']}: {record['details']}"
                )

            return "\n".join(context_parts)

    def upsert_communities(self, document_id: str, community_reports: Dict[int, Dict[str, Any]]):
        """Store community summaries and attach entity links."""
        from concurrent.futures import ThreadPoolExecutor

        def prepare_community(item):
            comm_id, data = item
            report = str(data.get("report", "")).strip()
            emb = self.embedding_service.embed_query(report)
            vector = [float(x) for x in emb]
            return {
                "comm_id": comm_id,
                "unique_comm_id": f"{document_id}_comm_{comm_id}",
                "report": report,
                "embedding": vector,
                "document_id": str(document_id),
                "nodes": data.get("nodes", []),
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            community_data = list(executor.map(prepare_community, community_reports.items()))

        with self.driver.session() as session:
            session.run(
                """
                UNWIND $batch AS data
                MERGE (c:Community {id: data.unique_comm_id})
                SET c.report = data.report,
                    c.embedding = data.embedding,
                    c.document_id = coalesce(c.document_id, data.document_id),
                    c.document_ids = CASE
                        WHEN c.document_ids IS NULL THEN [data.document_id]
                        WHEN data.document_id IN c.document_ids THEN c.document_ids
                        ELSE c.document_ids + data.document_id
                    END,
                    c.local_id = data.comm_id
                """,
                batch=community_data,
            )

            link_payloads = []
            for comm in community_data:
                for node_name in comm.get("nodes", []):
                    canonical_name = str(node_name).strip().lower()
                    if not canonical_name:
                        continue
                    link_payloads.append(
                        {
                            "entity_name": canonical_name,
                            "unique_comm_id": comm["unique_comm_id"],
                        }
                    )

            if link_payloads:
                session.run(
                    """
                    UNWIND $batch AS data
                    MATCH (e:Entity {name: data.entity_name})
                    MATCH (c:Community {id: data.unique_comm_id})
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                    """,
                    batch=link_payloads,
                )

        LOG.info(
            "[KnowledgeGraphService] Stored %d communities for document %s",
            len(community_reports),
            document_id,
        )

    @staticmethod
    def _create_community_node(tx, comm_id, report, embedding, doc_id):
        unique_comm_id = f"{doc_id}_comm_{comm_id}"
        query = """
        MERGE (c:Community {id: $unique_comm_id})
        SET c.report = $report,
            c.embedding = $embedding,
            c.document_id = coalesce(c.document_id, $doc_id),
            c.document_ids = CASE
                WHEN c.document_ids IS NULL THEN [$doc_id]
                WHEN $doc_id IN c.document_ids THEN c.document_ids
                ELSE c.document_ids + $doc_id
            END,
            c.local_id = $comm_id
        """
        tx.run(
            query,
            unique_comm_id=unique_comm_id,
            report=report,
            embedding=embedding,
            doc_id=str(doc_id),
            comm_id=comm_id,
        )

    @staticmethod
    def _link_entity_to_community(tx, comm_id, entity_name, doc_id):
        unique_comm_id = f"{doc_id}_comm_{comm_id}"
        query = """
        MATCH (e:Entity {name: $entity_name})
        MATCH (c:Community {id: $unique_comm_id})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        tx.run(query, entity_name=str(entity_name).strip().lower(), unique_comm_id=unique_comm_id)
