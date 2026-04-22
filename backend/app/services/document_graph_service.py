from __future__ import annotations

import logging
from typing import Any, Dict, List

import networkx as nx

from app.core.logger import LOG
from app.services.ner_service import NERService

logger = logging.getLogger(__name__)


class DocumentGraphService:
    """Builds in-memory knowledge graphs from document chunks."""

    def __init__(self, ner_service=None):
        self.ner_service = ner_service or NERService()
        # Legacy shared graph is kept for backward compatibility only.
        self.graph = nx.Graph()

    def build_in_memory_graph(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        graph: nx.Graph | None = None,
    ) -> nx.Graph:
        """
        Build a graph for one document.

        When `graph` is provided, merges into that graph.
        When omitted, creates a fresh graph (session-safe behavior).
        """
        LOG.info("Bat dau trich xuat do thi in-memory cho document_id=%s", document_id)
        target_graph = graph if graph is not None else nx.Graph()

        for idx, chunk in enumerate(chunks):
            text = chunk.get("page_content", "")
            if not text:
                continue

            LOG.info("Dang xu ly chunk %s/%s", idx + 1, len(chunks))
            graph_data = self.ner_service.extract_graph_elements(text)

            for entity in graph_data.get("entities", []):
                node_name = str(entity.get("name", "")).strip().lower()
                if not node_name:
                    continue

                if not target_graph.has_node(node_name):
                    target_graph.add_node(
                        node_name,
                        original_name=entity.get("name"),
                        type=entity.get("type", "Unknown"),
                        description=entity.get("description", ""),
                        document_ids={document_id},
                    )
                else:
                    target_graph.nodes[node_name].setdefault("document_ids", set()).add(document_id)

            for rel in graph_data.get("relationships", []):
                source = str(rel.get("source", "")).strip().lower()
                target = str(rel.get("target", "")).strip().lower()
                if not source or not target:
                    continue

                if not target_graph.has_node(source):
                    target_graph.add_node(source, original_name=rel.get("source"), type="Unknown", description="")
                if not target_graph.has_node(target):
                    target_graph.add_node(target, original_name=rel.get("target"), type="Unknown", description="")

                target_graph.add_edge(
                    source,
                    target,
                    relation=rel.get("relation", ""),
                    description=rel.get("description", ""),
                    document_id=document_id,
                )

        LOG.info(
            "Hoan thanh do thi cho %s: nodes=%s edges=%s",
            document_id,
            target_graph.number_of_nodes(),
            target_graph.number_of_edges(),
        )
        return target_graph

    def get_neighbors_context(self, entity_name: str) -> str:
        node = entity_name.strip().lower()
        if not self.graph.has_node(node):
            return ""

        neighbors = list(self.graph.neighbors(node))
        context = f"Thong tin ve '{entity_name}':\n"

        for neighbor in neighbors:
            edge_data = self.graph.get_edge_data(node, neighbor)
            relation = edge_data.get("relation", "")
            desc = edge_data.get("description", "")
            context += f"- Co quan he [{relation}] voi '{neighbor}': {desc}\n"

        return context
