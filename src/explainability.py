from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


@dataclass(frozen=True)
class PairExplanation:
    """Explainability payload for a candidate recommendation/link (u -> v)."""

    source: Any
    target: Any
    score: Optional[float]
    signals: Dict[str, Any]
    evidence: Dict[str, Any]


def _safe_first(generator: Iterable[Tuple[Any, Any, float]], default: float = 0.0) -> float:
    for _, _, p in generator:
        return float(p)
    return float(default)


def explain_pair(
    G: nx.Graph,
    source: Any,
    target: Any,
    *,
    score: Optional[float] = None,
    max_common_neighbors: int = 10,
    max_path_len: int = 4,
) -> PairExplanation:
    """Build a human-friendly explanation for why `target` is recommended for `source`.

    This is designed for a graph-based product graph (co-purchase/co-view). It uses
    transparent signals:
    - common neighbors ("cùng được mua với")
    - Jaccard / Adamic-Adar / Preferential Attachment
    - shortest path evidence (2-4 hops)
    - optional node attributes if present: community, pagerank, degree_centrality
    """

    if source not in G or target not in G:
        raise ValueError("source/target must exist in graph")

    # Neighborhood evidence
    common_neighbors = list(nx.common_neighbors(G, source, target))
    common_neighbors_preview = common_neighbors[:max_common_neighbors]

    # Heuristic similarity signals
    jaccard = _safe_first(nx.jaccard_coefficient(G, [(source, target)]))
    try:
        adamic_adar = _safe_first(nx.adamic_adar_index(G, [(source, target)]))
    except nx.NetworkXError:
        adamic_adar = 0.0
    pref_attach = _safe_first(nx.preferential_attachment(G, [(source, target)]))

    # Path evidence (short, easy-to-explain)
    path: Optional[List[Any]]
    path_len: Optional[int]
    try:
        path = nx.shortest_path(G, source=source, target=target)
        path_len = len(path) - 1
        if path_len > max_path_len:
            # Too long to be a good explanation; keep only length.
            path = None
    except nx.NetworkXNoPath:
        path = None
        path_len = None

    # Optional attributes if you ran Louvain/Centrality scripts
    src_attrs = G.nodes[source]
    tgt_attrs = G.nodes[target]
    src_comm = src_attrs.get("community")
    tgt_comm = tgt_attrs.get("community")
    same_community = (src_comm is not None) and (tgt_comm is not None) and (src_comm == tgt_comm)

    signals: Dict[str, Any] = {
        "has_direct_edge": bool(G.has_edge(source, target)),
        "common_neighbors_count": int(len(common_neighbors)),
        "jaccard": float(jaccard),
        "adamic_adar": float(adamic_adar),
        "preferential_attachment": float(pref_attach),
        "shortest_path_len": path_len,
        "same_community": bool(same_community),
        "source_degree": int(G.degree(source)),
        "target_degree": int(G.degree(target)),
        "source_pagerank": src_attrs.get("pagerank"),
        "target_pagerank": tgt_attrs.get("pagerank"),
        "source_degree_centrality": src_attrs.get("degree_centrality"),
        "target_degree_centrality": tgt_attrs.get("degree_centrality"),
        "source_community": src_comm,
        "target_community": tgt_comm,
    }

    evidence: Dict[str, Any] = {
        "common_neighbors": common_neighbors_preview,
        "shortest_path": path,
    }

    return PairExplanation(
        source=source,
        target=target,
        score=score,
        signals=signals,
        evidence=evidence,
    )


def explanation_to_vi_text(exp: PairExplanation) -> str:
    """Format explanation to short Vietnamese text (for UI/API response)."""

    s = exp.signals
    e = exp.evidence

    parts: List[str] = []
    if exp.score is not None:
        parts.append(f"Điểm gợi ý: {exp.score:.4f}.")

    cn = s.get("common_neighbors_count", 0)
    if cn:
        preview = e.get("common_neighbors") or []
        preview_txt = ", ".join(map(str, preview[:5]))
        suffix = f" (ví dụ: {preview_txt})" if preview else ""
        parts.append(f"Có {cn} sản phẩm liên quan chung (common neighbors){suffix}.")

    if s.get("same_community"):
        parts.append(f"Cùng community (Louvain) #{s.get('source_community')}." )

    j = s.get("jaccard")
    aa = s.get("adamic_adar")
    if j is not None or aa is not None:
        parts.append(f"Tương đồng cấu trúc: Jaccard={float(j):.4f}, Adamic-Adar={float(aa):.4f}.")

    path = e.get("shortest_path")
    if path:
        parts.append("Đường liên hệ ngắn: " + " → ".join(map(str, path)) + ".")

    pr = s.get("target_pagerank")
    if pr is not None:
        parts.append(f"Target có PageRank cao (pagerank={float(pr):.6f}).")

    if not parts:
        return "Chưa đủ tín hiệu để tạo giải thích ngắn gọn (thiếu community/centrality/neighbor evidence)."
    return " ".join(parts)
