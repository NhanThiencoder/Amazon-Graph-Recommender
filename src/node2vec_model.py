# src/node2vec_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Node2VecConfig:
    dimensions:   int = 32
    walk_length:  int = 10   # không dùng, giữ để interface không đổi
    num_walks:    int = 3    # không dùng, giữ để interface không đổi
    epochs:       int = 1    # không dùng, giữ để interface không đổi
    batch_size:   int = 256  # không dùng, giữ để interface không đổi
    seed:         int = 42


def fit_node2vec_torch(
    G: nx.Graph,
    config: Node2VecConfig,
) -> Dict[Any, Sequence[float]]:
    """
    Tạo node embeddings bằng SVD trên adjacency matrix.
    Không cần PyTorch hay gensim — chỉ dùng numpy + scipy.
    Interface giữ nguyên: trả về dict {node: embedding_vector}.
    """
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Xây adjacency matrix dạng sparse
    rows, cols = [], []
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # SVD rút gọn → lấy k chiều đầu làm embedding
    k = min(config.dimensions, n - 1)
    np.random.seed(config.seed)
    U, S, _ = spla.svds(A, k=k, random_state=config.seed)

    # Nhân với căn bậc hai singular values để scale đúng
    embeddings_matrix = U * np.sqrt(S)

    return {node: embeddings_matrix[node_index[node]] for node in nodes}