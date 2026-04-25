from __future__ import annotations

import os
from typing import Any, Dict, List

from app.core.logger import LOG
from app.core.config import settings
from app.services.detect_community_service import DetectCommunityService
from app.services.document_graph_service import DocumentGraphService
from app.services.document_service import DocumentService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.ner_service import NERService


class GraphRAGService:
    """
    Session-safe GraphRAG indexing service.

    Important behavior:
    - Builds graph per document independently.
    - Does NOT keep a global in-memory graph across sessions.
    - Neo4j storage remains shared, but retrieval is filtered by session documents.
    """

    def __init__(self, embedding_service=None):
        self.ner_service = NERService()
        self.doc_service = DocumentService()
        self.doc_graph_service = DocumentGraphService(ner_service=self.ner_service)
        self.community_service = DetectCommunityService(ner_service=self.ner_service)
        self.kg_service = KnowledgeGraphService(embedding_service=embedding_service)

    def _get_original_filename(self, filename: str) -> str:
        parts = filename.split("_", 3)
        if len(parts) >= 4:
            return parts[3]
        return filename

    @staticmethod
    def _to_chunk_dicts(langchain_docs) -> List[Dict[str, Any]]:
        return [{"page_content": doc.page_content, **(doc.metadata or {})} for doc in langchain_docs]

    @staticmethod
    def _nx_graph_to_dict(nx_graph) -> Dict[str, Any]:
        graph_data: Dict[str, Any] = {"entities": [], "relationships": []}

        for node, data in nx_graph.nodes(data=True):
            graph_data["entities"].append(
                {
                    "name": node,
                    "type": data.get("type", "Unknown"),
                    "description": data.get("description", ""),
                }
            )

        for u, v, data in nx_graph.edges(data=True):
            graph_data["relationships"].append(
                {
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", "RELATED"),
                    "description": data.get("description", ""),
                    "confidence": float(data.get("confidence", 0.0) or 0.0),
                }
            )

        return graph_data

    def build_graph_for_document(self, document_id: int, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Khong tim thay file: {file_path}")

        doc_id_str = str(document_id)
        display_name = self._get_original_filename(os.path.basename(file_path))
        LOG.info("[GraphRAG] Building graph for doc_id=%s (%s)", doc_id_str, display_name)

        langchain_docs = self.doc_service.load_document(file_path)
        if not langchain_docs:
            raise ValueError(f"Tai lieu '{display_name}' khong co noi dung de xay do thi")

        chunk_dicts = self._to_chunk_dicts(langchain_docs)
        nx_graph = self.doc_graph_service.build_in_memory_graph(doc_id_str, chunk_dicts)

        graph_data = self._nx_graph_to_dict(nx_graph)
        self.kg_service.upsert_graph(doc_id_str, graph_data)

        community_reports = self.community_service.detect_and_summarize(nx_graph)
        self.kg_service.upsert_communities(doc_id_str, community_reports)

        LOG.info(
            "[GraphRAG] Completed graph for doc_id=%s: nodes=%s edges=%s communities=%s",
            doc_id_str,
            nx_graph.number_of_nodes(),
            nx_graph.number_of_edges(),
            len(community_reports),
        )

    def update_graph_with_documents(self, documents: List[Dict[str, Any]]):
        for document in documents:
            document_id = int(document["id"])
            filepath = str(document["filepath"])
            self.build_graph_for_document(document_id, filepath)

    def update_graph_with_file(self, file_path: str):
        from app.services.database_service import db_service

        doc_record = db_service.get_document_by_filepath(file_path)
        if not doc_record:
            raise ValueError(f"Khong tim thay document record cho file: {file_path}")
        self.build_graph_for_document(int(doc_record["id"]), str(file_path))

    def is_graph_ready(self) -> bool:
        return self.kg_service.check_if_graph_exists()

    def is_document_indexed(self, document_id: int) -> bool:
        return self.kg_service.is_document_indexed(str(document_id))

    def are_documents_indexed(self, document_ids: List[int]) -> bool:
        if not document_ids:
            return False
        return all(self.is_document_indexed(int(doc_id)) for doc_id in document_ids)

    def get_new_files(self) -> List[str]:
        doc_dir = settings.DOCUMENT_DIR
        if not os.path.exists(doc_dir):
            return []

        all_files = [
            f
            for f in os.listdir(doc_dir)
            if f.lower().endswith((".pdf", ".docx")) and os.path.isfile(os.path.join(doc_dir, f))
        ]

        indexed_ids = set(self.kg_service.get_indexed_document_ids())
        return [os.path.join(doc_dir, f) for f in all_files if f not in indexed_ids]
