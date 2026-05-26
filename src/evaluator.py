from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import math                          
import networkx as nx                # thư viện xử lý đồ thị
import community as community_louvain  
from collections import defaultdict  

Edge = Tuple[Any, Any]  # kiểu alias cho 1 cặp node (u, v)

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: HÀM TÍNH ĐIỂM TỪ EMBEDDING (dùng cho Node2Vec)
# ══════════════════════════════════════════════════════════════════════════════

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Tích vô hướng (dot product) giữa 2 vector."""
    return float(sum(x * y for x, y in zip(a, b)))

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Độ tương đồng cosine giữa 2 vector — kết quả trong [-1, 1]."""
    da = math.sqrt(sum(x * x for x in a))  
    db = math.sqrt(sum(x * x for x in b))
    if da == 0.0 or db == 0.0:
        return 0.0                          
    return _dot(a, b) / (da * db)           # công thức cosine chuẩn


ScoreFn = Literal["dot", "cosine"]  # chỉ chấp nhận 2 kiểu tính điểm này


def score_edges_from_embeddings(
    embeddings: Dict[Any, Sequence[float]],  
    edges: Iterable[Edge],       
    *,
    method: ScoreFn = "dot",                 # phương pháp tính: dot hoặc cosine
) -> List[float]:
    """
    Tính điểm dự đoán liên kết cho mỗi cặp node dựa trên embedding.
    Node nào không có embedding sẽ nhận điểm 0.
    """
    scorer = _dot if method == "dot" else _cosine  # chọn hàm tính điểm

    scores: List[float] = []  # danh sách điểm đầu ra

    for u, v in edges:
        eu = embeddings.get(u)  
        ev = embeddings.get(v)  

        if eu is None or ev is None:
            scores.append(0.0)              # node chưa được embed → điểm = 0
        else:
            scores.append(float(scorer(eu, ev)))  # tính độ tương đồng 2 vector

    return scores

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: ĐÁNH GIÁ LINK PREDICTION (Precision, Recall, AUC, AP)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class LinkPredictionReport:
    """Chứa toàn bộ chỉ số đánh giá của 1 thuật toán link prediction."""
    auc:       float  # diện tích dưới đường ROC — càng gần 1 càng tốt
    ap:        float  # average precision — càng gần 1 càng tốt
    precision: float  
    recall:    float  
    f1:        float  
    threshold: float  # ngưỡng phân loại tối ưu (tự tìm theo F1)


def evaluate_link_prediction(
    pos_scores: Sequence[float],   # điểm của các cạnh THẬT bị giấu
    neg_scores: Sequence[float],   # điểm của các cặp GIẢ (không có cạnh)
    *,
    threshold: Optional[float] = None,  # None = tự tìm ngưỡng tốt nhất
) -> LinkPredictionReport:
    """
    Tính AUC, Average Precision, Precision, Recall, F1 cho 1 thuật toán.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        precision_recall_curve,
    )
    import numpy as np

    # Tạo nhãn: 1 = cạnh thật, 0 = cặp giả
    y_true  = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_score = list(pos_scores) + list(neg_scores)  # ghép điểm của 2 nhóm lại

    if len(set(y_true)) < 2:
        raise ValueError("Cần có cả positive (cạnh thật) và negative (cặp giả)")

    auc = float(roc_auc_score(y_true, y_score))           # tính AUC
    ap  = float(average_precision_score(y_true, y_score)) # tính Average Precision

    # Tự tìm threshold tối ưu theo F1 nếu người dùng không truyền vào
    if threshold is None:
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_score)
        f1_arr   = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)  # F1 tại mỗi ngưỡng
        best_idx  = int(np.argmax(f1_arr))                                # vị trí F1 cao nhất
        threshold = float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else 0.5

    # Chuyển điểm sang nhãn 0/1 theo ngưỡng vừa tìm được
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    prec = float(precision_score(y_true, y_pred, zero_division=0))  # Precision
    rec  = float(recall_score(y_true, y_pred, zero_division=0))     # Recall
    f1   = float(f1_score(y_true, y_pred, zero_division=0))         # F1-score

    return LinkPredictionReport(
        auc=auc, ap=ap,
        precision=prec, recall=rec, f1=f1,
        threshold=threshold,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: HÀM TỔNG HỢP — STREAMLIT GỌI HÀM NÀY ĐỂ LẤY MỌI KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelComparisonResult:
    """
    Đóng gói toàn bộ kết quả Giai đoạn 3.
    Streamlit chỉ cần gọi run_model_comparison() một lần rồi dùng object này.
    """
    report_jaccard:  LinkPredictionReport        # kết quả Jaccard
    report_adamic:   LinkPredictionReport        # kết quả Adamic-Adar
    report_node2vec: Optional[LinkPredictionReport]  # kết quả Node2Vec (None nếu tắt)

    # Điểm thô của từng thuật toán — dùng để vẽ ROC curve
    jacc_pos: List[float]   
    jacc_neg: List[float]   
    adam_pos: List[float]  
    adam_neg: List[float]   
    n2v_pos:  List[float]   
    n2v_neg:  List[float] 

    # Chỉ số mạng lưới
    modularity:      Optional[float]  
    avg_clustering:  float            
    num_communities: int            

    # Thông tin về tập dữ liệu sau khi chia
    n_nodes_train:   int  
    n_edges_train:   int  
    n_pos_test:      int  
    n_neg_test:      int  

    # Thông tin về Node2Vec
    node2vec_used_subgraph: bool  
    node2vec_subgraph_size: int  


def run_model_comparison(
    G: nx.Graph,             
    *,
    test_frac: float = 0.20,  
    val_frac:  float = 0.10,  
    seed:      int   = 42,    

    # Tham số Node2Vec
    run_node2vec:           bool = True,  
    node2vec_subgraph_size: int  = 5000,   
    node2vec_dimensions:    int  = 32,    
    node2vec_walk_length:   int  = 10,     
    node2vec_num_walks:     int  = 3,    
    node2vec_epochs:        int  = 1,    
    node2vec_batch_size:    int  = 256, 
) -> ModelComparisonResult:
    """
    Thứ tự thực hiện:
      1. Ẩn 20% cạnh (edge split)
      2. Dự đoán bằng Jaccard + Adamic-Adar
      3. Dự đoán bằng Node2Vec PyTorch (trên subgraph nhỏ)
      4. Tính Precision, Recall, AUC, AP cho từng thuật toán
      5. Tính Modularity + Clustering Coefficient

    """
    # Import các module trong src/ — thử cả 2 cách import để tương thích
    try:
        from src.edge_split          import train_val_test_edge_split
        from src.link_prediction     import predict_links
        from src.node2vec_model      import fit_node2vec_torch, Node2VecConfig
        from src.community_detection import run_louvain_clustering
    except ImportError:
        from .edge_split          import train_val_test_edge_split
        from .link_prediction     import predict_links
        from .node2vec_model      import fit_node2vec_torch, Node2VecConfig
        from .community_detection import run_louvain_clustering

    # ── BƯỚC 1: Ẩn 20% cạnh ──────────────────────────────────────────────────
    split = train_val_test_edge_split(
        G, test_frac=test_frac, val_frac=val_frac, seed=seed
    )

    # Gộp cạnh thật bị giấu + cặp giả thành 1 danh sách để predict cùng lúc
    all_test_pairs = split.pos_test + split.neg_test
    n_pos = len(split.pos_test)  

    # ── BƯỚC 2: Dự đoán bằng Jaccard và Adamic-Adar ──────────────────────────
    df_preds = predict_links(split.G_train, all_test_pairs)  # trả về DataFrame

    # Tách điểm Jaccard cho cạnh thật và cặp giả
    jacc_pos = df_preds["Jaccard_Score"].iloc[:n_pos].tolist()   
    jacc_neg = df_preds["Jaccard_Score"].iloc[n_pos:].tolist()    

    # Tách điểm Adamic-Adar tương tự
    adam_pos = df_preds["Adamic_Adar_Score"].iloc[:n_pos].tolist()
    adam_neg = df_preds["Adamic_Adar_Score"].iloc[n_pos:].tolist()

    # Tính Precision, Recall, AUC, AP cho Jaccard và Adamic-Adar
    report_jaccard = evaluate_link_prediction(jacc_pos, jacc_neg)
    report_adamic  = evaluate_link_prediction(adam_pos, adam_neg)

    # ── BƯỚC 3: Dự đoán bằng Node2Vec (PyTorch baseline) ─────────────────────
    n2v_pos: List[float] = []   # khởi tạo rỗng phòng trường hợp tắt Node2Vec
    n2v_neg: List[float] = []
    report_node2vec: Optional[LinkPredictionReport] = None
    used_subgraph = False       # flag ghi lại có dùng subgraph không

    if run_node2vec:
        G_train = split.G_train

        # Nếu graph quá lớn → lấy subgraph gồm các node có degree cao nhất
        if G_train.number_of_nodes() > node2vec_subgraph_size:
            used_subgraph = True  # đánh dấu đã dùng subgraph
            top_nodes    = sorted(G_train.degree, key=lambda x: x[1], reverse=True)[:node2vec_subgraph_size]
            sub_node_ids = [n for n, _ in top_nodes]                # lấy id node
            G_n2v        = G_train.subgraph(sub_node_ids).copy()    # tạo subgraph
        else:
            G_n2v = G_train  # graph đủ nhỏ → dùng nguyên

        # Cấu hình nhẹ để tiết kiệm RAM và thời gian
        cfg = Node2VecConfig(
            dimensions=node2vec_dimensions,    # số chiều vector
            walk_length=node2vec_walk_length,  # độ dài walk
            num_walks=node2vec_num_walks,      # số walk/node
            epochs=node2vec_epochs,            # số epoch
            batch_size=node2vec_batch_size,    # batch size
            seed=seed,
        )

        embeddings = fit_node2vec_torch(G_n2v, config=cfg)  # train và lấy embedding

        # Tính điểm cosine similarity cho cạnh thật và cặp giả
        n2v_pos = score_edges_from_embeddings(embeddings, split.pos_test, method="cosine")
        n2v_neg = score_edges_from_embeddings(embeddings, split.neg_test, method="cosine")

        report_node2vec = evaluate_link_prediction(n2v_pos, n2v_neg)  # tính các chỉ số

    # ── BƯỚC 4: Modularity + Clustering Coefficient ───────────────────────────
    topology = evaluate_network_topology(G, partition=None)  # khởi tạo không có partition
    num_communities = 0  

    try:
        _, df_comm = run_louvain_clustering(G) 

        # Chuyển DataFrame thành dict {node: community_id} để tính Modularity
        partition = {
            int(row["ProductID"]): int(row["CommunityID"])
            for _, row in df_comm.iterrows()
        }

        topology        = evaluate_network_topology(G, partition=partition)  
        num_communities = int(df_comm["CommunityID"].nunique())  
    except Exception:
        pass 

    # ── TRẢ VỀ KẾT QUẢ ────────────────────────────────────────────────────────
    return ModelComparisonResult(
        report_jaccard=report_jaccard,
        report_adamic=report_adamic,
        report_node2vec=report_node2vec,
        jacc_pos=jacc_pos, jacc_neg=jacc_neg,
        adam_pos=adam_pos, adam_neg=adam_neg,
        n2v_pos=n2v_pos,   n2v_neg=n2v_neg,
        modularity=topology.get("modularity"),        
        avg_clustering=topology.get("avg_clustering", 0.0), 
        num_communities=num_communities,
        n_nodes_train=split.G_train.number_of_nodes(),
        n_edges_train=split.G_train.number_of_edges(),
        n_pos_test=len(split.pos_test),  
        n_neg_test=len(split.neg_test),  
        node2vec_used_subgraph=used_subgraph,
        node2vec_subgraph_size=node2vec_subgraph_size,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: ĐÁNH GIÁ MẠNG LƯỚI (Modularity + Clustering Coefficient)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_network_topology(
    G: nx.Graph,        # đồ thị cần đánh giá
    partition=None,     # dict {node: community_id} từ Louvain — None thì bỏ qua Modularity
) -> Dict[str, Any]:
    """
    Tính 2 chỉ số cấu trúc mạng lưới:
      - avg_clustering : Average Clustering Coefficient
      - modularity     : Modularity Score (chỉ tính nếu có partition)
    """
    results: Dict[str, Any] = {}

    # Clustering Coefficient — đo mức độ "bạn của bạn cũng là bạn nhau"
    results["avg_clustering"] = nx.average_clustering(G)  # giá trị từ 0 đến 1

    if partition is not None:
        # Chuyển dict {node: community_id} thành list các set
        community_dict = defaultdict(set)
        for node, com_id in partition.items():
            community_dict[com_id].add(node)  # gom các node cùng cộng đồng vào 1 set

        communities_list = list(community_dict.values())  # list các set node

        # Tính Modularity — đo mức độ tách biệt giữa các cộng đồng (> 0.3 là tốt)
        try:
            results["modularity"] = nx.community.modularity(G, communities_list)  # NetworkX mới
        except AttributeError:
            import networkx.algorithms.community as nx_comm
            results["modularity"] = nx_comm.modularity(G, communities_list)       # NetworkX cũ
    else:
        results["modularity"] = None  # không có partition → không tính được Modularity

    return results