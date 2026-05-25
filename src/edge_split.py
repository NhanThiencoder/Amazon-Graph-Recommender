from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import random

import networkx as nx


Edge = Tuple[Any, Any]


@dataclass(frozen=True)
class EdgeSplit:
    """Container for a standard link-prediction split."""

    G_train: nx.Graph
    pos_train: List[Edge]
    pos_val: List[Edge]
    pos_test: List[Edge]
    neg_val: List[Edge]
    neg_test: List[Edge]


def _normalize_undirected_edge(u: Any, v: Any) -> Edge:
    return (u, v) if u <= v else (v, u)


def sample_non_edges(
    G: nx.Graph,
    num_samples: int,
    *,
    seed: int = 42,
    forbidden: Sequence[Edge] | None = None,
) -> List[Edge]:
    """Sample random node pairs that are NOT edges in G."""

    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []

    forbidden_set = set()
    if forbidden:
        forbidden_set = {_normalize_undirected_edge(u, v) for u, v in forbidden}

    out: set[Edge] = set()
    # loop with cap to avoid infinite in dense graphs
    cap = max(10_000, num_samples * 50)
    tries = 0
    while len(out) < num_samples and tries < cap:
        u, v = rng.sample(nodes, 2)
        if u == v:
            tries += 1
            continue
        a, b = _normalize_undirected_edge(u, v)
        if (a, b) in forbidden_set:
            tries += 1
            continue
        if G.has_edge(a, b):
            tries += 1
            continue
        out.add((a, b))
        tries += 1
    return list(out)


def train_val_test_edge_split(
    G: nx.Graph,
    *,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
    ensure_connected: bool = True,
) -> EdgeSplit:
    """Hide edges for link prediction.

    - Build G_train by removing val/test positive edges.
    - If ensure_connected=True, avoid removing bridges so G_train stays connected.
    - Negative edges are sampled from non-edges of the *original* graph node set.
    """

    if not (0.0 < test_frac < 1.0) or not (0.0 <= val_frac < 1.0):
        raise ValueError("test_frac and val_frac must be in (0,1)")

    rng = random.Random(seed)

    # Work on a copy
    G0 = G.copy()
    if G0.is_directed():
        raise ValueError("This splitter expects an undirected graph")

    edges = list(G0.edges())
    if not edges:
        raise ValueError("Graph has no edges")

    # Normalize edges for stable sets
    edges_norm = [_normalize_undirected_edge(u, v) for u, v in edges]
    edges_norm = list(set(edges_norm))
    rng.shuffle(edges_norm)

    n_total = len(edges_norm)
    n_test = max(1, int(n_total * test_frac))
    n_val = max(0, int(n_total * val_frac))
    n_holdout = min(n_total - 1, n_test + n_val)  # keep at least 1 edge

    holdout_candidates = edges_norm[: n_holdout * 3]  # allow retries if connected constraint

    pos_test: List[Edge] = []
    pos_val: List[Edge] = []
    removed: List[Edge] = []

    # Precompute bridges if needed
    bridges: set[Edge] = set()
    if ensure_connected:
        try:
            bridges = {_normalize_undirected_edge(u, v) for u, v in nx.bridges(G0)}
        except Exception:
            bridges = set()

    # Select holdout edges
    for e in holdout_candidates:
        if len(pos_test) >= n_test and len(pos_val) >= n_val:
            break
        if ensure_connected and e in bridges:
            continue
        removed.append(e)
        if len(pos_test) < n_test:
            pos_test.append(e)
        else:
            pos_val.append(e)

    if len(pos_test) < n_test:
        raise RuntimeError("Could not sample enough test edges without disconnecting graph")

    # Build G_train
    G_train = G0.copy()
    G_train.remove_edges_from(removed)

    # If connectivity still broke (bridges estimate was approximate), revert constraint.
    if ensure_connected and not nx.is_connected(G_train):
        # Fallback: no connectivity constraint
        G_train = G0.copy()
        pos_test = edges_norm[:n_test]
        pos_val = edges_norm[n_test : n_test + n_val]
        removed = pos_test + pos_val
        G_train.remove_edges_from(removed)

    # Training positives are remaining edges
    pos_train = [_normalize_undirected_edge(u, v) for u, v in G_train.edges()]
    pos_train = list(set(pos_train))

    # Negatives sampled from non-edges in the ORIGINAL graph (avoid true edges)
    forbidden = edges_norm
    neg_val = sample_non_edges(G0, len(pos_val), seed=seed + 1, forbidden=forbidden)
    neg_test = sample_non_edges(G0, len(pos_test), seed=seed + 2, forbidden=forbidden)

    return EdgeSplit(
        G_train=G_train,
        pos_train=pos_train,
        pos_val=pos_val,
        pos_test=pos_test,
        neg_val=neg_val,
        neg_test=neg_test,
    )
