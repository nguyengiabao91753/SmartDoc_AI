from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
from graspologic.partition import hierarchical_leiden

from app.ai.llm import get_llm
from app.core.logger import LOG
from app.services.ner_service import NERService


class DetectCommunityService:
    """Detect graph communities and generate compact summaries for GraphRAG."""

    def __init__(self, ner_service=None):
        self.ner_service = ner_service or NERService()
        self.llm = get_llm(temperature=0.0)

    def detect_and_summarize(self, graph: nx.Graph) -> Dict[int, Dict[str, Any]]:
        return self.detect_and_summarize_with_progress(graph, progress_callback=None)

    def detect_and_summarize_with_progress(
        self,
        graph: nx.Graph,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        def _notify(detail: str, pct: int = 0):
            if progress_callback:
                progress_callback(detail, pct)

        if graph.number_of_nodes() == 0:
            return {}

        isolates = list(nx.isolates(graph))
        if isolates:
            graph.remove_nodes_from(isolates)

        _notify("Dang chay thuat toan Leiden...", 10)
        community_mapping = hierarchical_leiden(graph, max_cluster_size=10)

        communities: Dict[int, List[str]] = {}
        for partition in community_mapping:
            communities.setdefault(partition.cluster, []).append(partition.node)

        valid_communities = {cid: nodes for cid, nodes in communities.items() if len(nodes) >= 3}
        total_valid = len(valid_communities)

        community_reports: Dict[int, Dict[str, Any]] = {}
        if total_valid > 0:
            _notify(f"Bat dau tom tat {total_valid} cong dong...", 30)
            max_workers = 4

            def process_single_community(comm_id: int, node_list: List[str]):
                context = self._get_community_context(graph, node_list)
                summary = self._generate_report(comm_id, context, node_list)
                return comm_id, node_list, summary

            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_community, cid, nodes): cid
                    for cid, nodes in valid_communities.items()
                }

                for future in as_completed(futures):
                    try:
                        comm_id, node_list, summary = future.result()
                        community_reports[comm_id] = {"nodes": node_list, "report": summary}
                        completed += 1
                        _notify(
                            f"Da tom tat {completed}/{total_valid} cong dong...",
                            30 + int(70 * completed / total_valid),
                        )
                    except Exception as exc:
                        LOG.error("[Community] Failed to summarize cluster %s: %s", futures[future], exc)

        for cid, nodes in communities.items():
            if cid not in community_reports:
                community_reports[cid] = {
                    "nodes": nodes,
                    "report": self._fallback_summary(nodes),
                }

        _notify("Hoan tat xu ly do thi tri thuc.", 100)
        return community_reports

    def _get_community_context(self, graph: nx.Graph, node_list: List[str]) -> str:
        # Build richer context from node descriptions and edge descriptions.
        lines: List[str] = []

        for node in node_list[:20]:
            node_data = graph.nodes[node]
            entity_type = str(node_data.get("type", "Unknown"))
            desc = str(node_data.get("description", "")).strip()
            if desc:
                lines.append(f"ENTITY: {node} | type={entity_type} | desc={desc}")
            else:
                lines.append(f"ENTITY: {node} | type={entity_type}")

        subgraph = graph.subgraph(node_list)
        for u, v, data in list(subgraph.edges(data=True))[:40]:
            relation = str(data.get("relation", "RELATED"))
            rel_desc = str(data.get("description", "")).strip()
            if rel_desc:
                lines.append(f"REL: {u} -[{relation}]-> {v} | detail={rel_desc}")
            else:
                lines.append(f"REL: {u} -[{relation}]-> {v}")

        return "\n".join(lines)[:3000]

    def _generate_report(self, comm_id: int, context: str, node_list: List[str]) -> str:
        prompt = (
            "Ban dang tao bao cao cho 1 cong dong trong Knowledge Graph.\n"
            "Dua vao DATA ben duoi, viet 2-3 cau tieng Viet: \\n"
            "(1) Cong dong nay chu yeu noi ve chu de gi; \\n"
            "(2) cac thuc the/noi dung trung tam nao noi bat; \\n"
            "(3) neu du lieu qua mo ho, phai noi ro muc do khong chac chan.\n"
            "Khong viet chung chung kieu 'co moi quan he'.\n"
            "Khong chen markdown.\n\n"
            f"DATA:\n{context}\n\n"
            "BAO CAO:"
        )
        try:
            answer = str(self.llm.invoke(prompt)).strip()
            if answer:
                return answer
        except Exception as exc:
            LOG.warning("[Community] LLM report generation failed for cluster %s: %s", comm_id, exc)

        return self._fallback_summary(node_list)

    @staticmethod
    def _fallback_summary(node_list: List[str]) -> str:
        top_entities = ", ".join(node_list[:5]) if node_list else "khong ro"
        return (
            "Cong dong nay gom nhom thuc the lien quan den "
            f"{top_entities}. Du lieu hien tai chua du de mo ta chu de sau hon."
        )
