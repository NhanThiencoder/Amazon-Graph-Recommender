from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import random 
import torch
import networkx as nx  
Edge = Tuple[Any, Any]  

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: CẤU HÌNH NODE2VEC
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@dataclass(frozen=True)
class Node2VecConfig:
    """Tập hợp tất cả tham số của Node2Vec — truyền vào fit_node2vec_torch()."""
    dimensions:       int   = 64    # số chiều của vector embedding mỗi node
    walk_length:      int   = 20    # số bước trong mỗi random walk
    num_walks:        int   = 5     # số walk thực hiện trên mỗi node
    window_size:      int   = 5     # cửa sổ context trong skip-gram
    negative_samples: int   = 5     # số mẫu âm mỗi cặp dương trong loss
    p:                float = 1.0   # tham số return — nhỏ hơn = dễ quay lại node cũ
    q:                float = 1.0   # tham số in-out — nhỏ hơn = khám phá xa hơn
    lr:               float = 0.01  # learning rate của Adam optimizer
    epochs:           int   = 2     # số epoch train
    batch_size:       int   = 512   # số cặp (center, context) xử lý mỗi lần
    seed:             int   = 42    # seed để tái lập kết quả
    device:           str   = device

# ══════════════════════════════════════════════════════════════════════════
# PHẦN 2: ALIAS TABLE — LẤY MẪU O(1)
# ══════════════════════════════════════════════════════════════════════════
def _alias_sample(rng: random.Random, accept: List[float], alias: List[int]) -> int:
    """Lấy mẫu từ alias table với độ phức tạp O(1)."""
    i = rng.randrange(len(accept))   # chọn ngẫu nhiên 1 vị trí
    if rng.random() < accept[i]:
        return i                      # nhận vị trí đó với xác suất accept[i]
    return alias[i]                   # ngược lại lấy vị trí thay thế


def _build_alias_table(probs: Sequence[float]) -> Tuple[List[float], List[int]]:
    """
    Xây dựng alias table từ danh sách xác suất.
    Cho phép lấy mẫu theo phân phối bất kỳ với O(1) thay vì O(n).
    """
    n = len(probs)
    if n == 0:
        return [], []  # trả về rỗng nếu không có xác suất nào

    scaled = [p * n for p in probs]  
    accept = [0.0] * n             
    alias  = [0]   * n         

    small: List[int] = []  # các ô có xác suất < 1 (sau khi nhân n)
    large: List[int] = []  # các ô có xác suất >= 1

    for i, sp in enumerate(scaled):
        (small if sp < 1.0 else large).append(i)  # phân loại từng ô

    # Ghép ô nhỏ với ô lớn để cân bằng
    while small and large:
        s = small.pop()  
        l = large.pop()   
        accept[s] = scaled[s]          # xác suất chấp nhận của ô nhỏ
        alias[s]  = l                  # nếu không chấp nhận → lấy ô lớn
        scaled[l] = scaled[l] - (1.0 - scaled[s])   # giảm xác suất ô lớn
        (small if scaled[l] < 1.0 else large).append(l)  # phân loại lại

    for i in large + small:
        accept[i] = 1.0  # các ô còn lại → luôn chấp nhận
        alias[i]  = i

    return accept, alias


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: BIASED RANDOM WALKER
# ══════════════════════════════════════════════════════════════════════════════

class _BiasedRandomWalker:
    """
    Thực hiện random walk có hướng kiểu Node2Vec với 2 tham số p và q.
    - p nhỏ → dễ quay lại node vừa thăm (BFS-like)
    - q nhỏ → dễ đi xa hơn (DFS-like)
    """

    def __init__(self, G: nx.Graph, *, p: float, q: float, seed: int = 42):
        if G.is_directed():
            raise ValueError("Node2Vec baseline chỉ hỗ trợ undirected graph")

        self.G   = G
        self.p   = float(p)
        self.q   = float(q)
        self.rng = random.Random(seed) 

        # Cache danh sách hàng xóm để không gọi G.neighbors() lặp lại
        self.neighbors: Dict[Any, List[Any]] = {n: list(G.neighbors(n)) for n in G.nodes()}

        # Alias table cho bước đầu tiên (chưa có node trước)
        self.alias_nodes: Dict[Any, Tuple[List[float], List[int]]] = {}

        # Alias table cho bước thứ 2 trở đi (có node trước → tính p, q)
        self.alias_edges: Dict[Tuple[Any, Any], Tuple[List[float], List[int], List[Any]]] = {}

        self._preprocess_transition_probs()

    def _preprocess_transition_probs(self) -> None:
        """Tính trước alias table cho tất cả node và cạnh để walk nhanh hơn."""
        # Bước đầu tiên: chọn đồng đều giữa các hàng xóm
        for node, nbrs in self.neighbors.items():
            if not nbrs:
                self.alias_nodes[node] = ([], [])  # node cô lập → không có hàng xóm
                continue
            probs = [1.0 / len(nbrs)] * len(nbrs)        # xác suất đồng đều
            self.alias_nodes[node] = _build_alias_table(probs)

        # Bước tiếp theo: xác suất phụ thuộc vào node trước đó (src → dst)
        for src in self.G.nodes():
            for dst in self.neighbors.get(src, []):
                dst_nbrs = self.neighbors.get(dst, [])
                if not dst_nbrs:
                    self.alias_edges[(src, dst)] = ([], [], [])
                    continue

                unnorm: List[float] = []
                for x in dst_nbrs:
                    if x == src:
                        unnorm.append(1.0 / self.p)       # quay lại node trước → chia p
                    elif self.G.has_edge(x, src):
                        unnorm.append(1.0)                 # hàng xóm chung → xác suất 1
                    else:
                        unnorm.append(1.0 / self.q)        # node xa hơn → chia q

                s     = sum(unnorm) or 1.0                 # tổng để chuẩn hoá
                probs = [u / s for u in unnorm]            # chuẩn hoá thành phân phối xác suất
                accept, alias = _build_alias_table(probs)
                self.alias_edges[(src, dst)] = (accept, alias, dst_nbrs)

    def walk(self, start_node: Any, walk_length: int) -> List[Any]:
        """Thực hiện 1 random walk bắt đầu từ start_node."""
        walk = [start_node]  # khởi tạo walk với node đầu

        while len(walk) < walk_length:
            cur  = walk[-1]                        # node hiện tại
            nbrs = self.neighbors.get(cur, [])
            if not nbrs:
                break  # node cô lập → dừng walk

            if len(walk) == 1:
                # Bước đầu tiên: chọn đồng đều
                accept, alias = self.alias_nodes[cur]
                if not accept:
                    break
                idx = _alias_sample(self.rng, accept, alias)
                walk.append(nbrs[idx])
            else:
                # Bước tiếp theo: dùng alias table có tính p, q
                prev = walk[-2]
                accept, alias, dst_nbrs = self.alias_edges.get((prev, cur), ([], [], []))
                if not accept:
                    break
                idx = _alias_sample(self.rng, accept, alias)
                walk.append(dst_nbrs[idx])

        return walk

    def generate_walks(
        self,
        nodes: Sequence[Any],
        *,
        num_walks: int,
        walk_length: int,
    ) -> List[List[Any]]:
        """Sinh tất cả random walk cho toàn bộ node."""
        walks: List[List[Any]] = []
        nodes = list(nodes)

        for _ in range(num_walks):
            self.rng.shuffle(nodes)           # xáo trộn thứ tự node mỗi lần lặp
            for n in nodes:
                walks.append(self.walk(n, walk_length))  # walk từ mỗi node

        return walks

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: SKIP-GRAM PAIRS STREAMING (tiết kiệm RAM)
# ══════════════════════════════════════════════════════════════════════════════

def _iter_skipgram_pairs(
    walks: Iterable[List[Any]],   # danh sách các walk
    window_size: int,              # kích thước cửa sổ context
    node_to_idx: Dict[Any, int],   # ánh xạ node → index số nguyên
):
    w = int(window_size)

    for walk in walks:
        for i, center in enumerate(walk):
            if center not in node_to_idx:
                continue                             # bỏ qua node không có trong index

            c_idx = node_to_idx[center]              # index của node trung tâm
            left  = max(0, i - w)                    # giới hạn trái của cửa sổ
            right = min(len(walk), i + w + 1)        # giới hạn phải của cửa sổ

            for j in range(left, right):
                if j == i:
                    continue                         # bỏ qua chính nó
                ctx = walk[j]
                if ctx not in node_to_idx:
                    continue                         # bỏ qua node không có trong index
                yield (c_idx, node_to_idx[ctx])      # yield từng cặp (không lưu vào list)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: HÀM CHÍNH — TRAIN NODE2VEC VÀ TRẢ VỀ EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def fit_node2vec_torch(
    G: nx.Graph,                          # đồ thị đầu vào
    *,
    config: Node2VecConfig = Node2VecConfig(),  # cấu hình tham số
    nodes: Optional[Sequence[Any]] = None,      # None = dùng toàn bộ node trong G
    subgraph_size: Optional[int] = None,        # giới hạn số node để tránh MemoryError
) -> Dict[Any, List[float]]:
    """
    Train Node2Vec bằng PyTorch, trả về dict {node_id: embedding_vector}.

    Cách dùng cơ bản:
        cfg = Node2VecConfig(dimensions=32, walk_length=10, num_walks=3, epochs=1)
        embeddings = fit_node2vec_torch(G, config=cfg, subgraph_size=5000)

    Tham số subgraph_size:
        Nếu G có nhiều node hơn subgraph_size, tự động lấy subgraph
        gồm các node có degree cao nhất. Điều này giúp tránh MemoryError
        khi chạy trên graph Amazon 334k node.
    """
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F

    # Giới hạn subgraph nếu graph quá lớn
    if subgraph_size is not None and G.number_of_nodes() > subgraph_size:
        top_nodes    = sorted(G.degree, key=lambda x: x[1], reverse=True)[:subgraph_size]
        sub_node_ids = [n for n, _ in top_nodes]        # lấy id các node degree cao nhất
        G            = G.subgraph(sub_node_ids).copy()  # tạo subgraph độc lập

    # Lấy danh sách node cần embed
    if nodes is None:
        nodes = list(G.nodes())
    else:
        nodes = list(nodes)

    rng = random.Random(config.seed)  # random generator để tái lập

    # Tạo ánh xạ 2 chiều giữa node_id và index số nguyên
    node_to_idx: Dict[Any, int] = {n: i for i, n in enumerate(nodes)}  # node → index
    idx_to_node: List[Any]      = nodes                                  # index → node
    num_nodes = len(nodes)

    if num_nodes < 2:
        raise ValueError("Cần ít nhất 2 node để train")

    # ── Sinh random walks ─────────────────────────────────────────────────────
    walker = _BiasedRandomWalker(G, p=config.p, q=config.q, seed=config.seed)
    walks  = walker.generate_walks(
        nodes,
        num_walks=config.num_walks,
        walk_length=config.walk_length,
    )

    # ── Khởi tạo model PyTorch ────────────────────────────────────────────────
    device  = torch.device(config.device)  # cpu hoặc cuda

    emb_in  = nn.Embedding(num_nodes, config.dimensions, sparse=False).to(device)  # embedding input (center)
    emb_out = nn.Embedding(num_nodes, config.dimensions, sparse=False).to(device)  # embedding output (context)

    nn.init.xavier_uniform_(emb_in.weight)   # khởi tạo weight theo Xavier
    nn.init.xavier_uniform_(emb_out.weight)

    optimizer = torch.optim.Adam(
        list(emb_in.parameters()) + list(emb_out.parameters()),
        lr=config.lr,  # tốc độ học
    )

    # Phân phối unigram cho negative sampling (degree^0.75 theo paper gốc Word2Vec)
    degrees   = [float(G.degree(idx_to_node[i])) for i in range(num_nodes)]
    weights   = [math.pow(d if d > 0 else 1.0, 0.75) for d in degrees]  # degree^0.75
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    weights_t = weights_t / weights_t.sum()  # chuẩn hoá thành phân phối xác suất

    def neg_sample(batch_size: int, k: int) -> torch.Tensor:
        """Lấy k mẫu âm cho mỗi cặp dương trong batch."""
        return torch.multinomial(
            weights_t,
            num_samples=batch_size * k,
            replacement=True,
        ).view(batch_size, k)  # reshape thành (batch_size, k)

    batch_size = int(config.batch_size)
    k          = int(config.negative_samples)  # số mẫu âm mỗi cặp

    # ── Train theo từng epoch ────────────────────────────────────────────────
    for _epoch in range(int(config.epochs)):

        # Buffer để tích luỹ đủ batch_size cặp rồi mới tính gradient
        batch_centers:  List[int] = []
        batch_contexts: List[int] = []

        # _iter_skipgram_pairs là generator → không tạo list khổng lồ trong RAM
        pair_gen = _iter_skipgram_pairs(walks, config.window_size, node_to_idx)

        for c_idx, ctx_idx in pair_gen:
            batch_centers.append(c_idx)    # index node trung tâm
            batch_contexts.append(ctx_idx) # index node context

            if len(batch_centers) < batch_size:
                continue  # chưa đủ batch → tiếp tục tích luỹ

            # ── Forward pass ─────────────────────────────────────────────────
            centers  = torch.tensor(batch_centers,  dtype=torch.long, device=device)
            contexts = torch.tensor(batch_contexts, dtype=torch.long, device=device)
            negs     = neg_sample(len(batch_centers), k)  # (B, K) mẫu âm

            v      = emb_in(centers)    # (B, D) — vector của node trung tâm
            u_pos  = emb_out(contexts)  # (B, D) — vector của node context (dương)
            u_neg  = emb_out(negs)      # (B, K, D) — vector của node âm

            # Negative sampling loss (theo công thức Word2Vec)
            pos_score = (v * u_pos).sum(dim=1)                    # (B,) dot product dương
            pos_loss  = F.logsigmoid(pos_score).mean()            # log σ(v·u_pos)

            neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (B, K) dot product âm
            neg_loss  = F.logsigmoid(-neg_score).mean()              # log σ(-v·u_neg)

            loss = -(pos_loss + neg_loss)  # tổng loss (negative vì muốn maximize)

            # ── Backward pass ─────────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)  # xoá gradient cũ
            loss.backward()                         # tính gradient
            optimizer.step()                        # cập nhật weight

            # Reset buffer sau mỗi batch
            batch_centers  = []
            batch_contexts = []

        # Xử lý batch cuối còn thừa (chưa đủ batch_size)
        if batch_centers:
            centers  = torch.tensor(batch_centers,  dtype=torch.long, device=device)
            contexts = torch.tensor(batch_contexts, dtype=torch.long, device=device)
            negs     = neg_sample(len(batch_centers), k)

            v         = emb_in(centers)
            u_pos     = emb_out(contexts)
            u_neg     = emb_out(negs)

            pos_score = (v * u_pos).sum(dim=1)
            pos_loss  = F.logsigmoid(pos_score).mean()
            neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
            neg_loss  = F.logsigmoid(-neg_score).mean()
            loss      = -(pos_loss + neg_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # ── Xuất embedding ────────────────────────────────────────────────────────
    vectors = emb_in.weight.detach().cpu().numpy()  # lấy ma trận embedding từ GPU về CPU

    # Trả về dict {node_id: vector dạng list} để dễ dùng với mọi thư viện
    return {
        idx_to_node[i]: vectors[i].astype(float).tolist()
        for i in range(num_nodes)
    }