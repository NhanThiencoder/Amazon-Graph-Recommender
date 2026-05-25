import networkx as nx
import pandas as pd
import time
import os
import pickle

def predict_links(G, candidate_pairs):
    """
    Dự đoán liên kết bằng Jaccard và Adamic-Adar cho tập các cặp ứng viên.
    :param G: Đồ thị NetworkX
    :param candidate_pairs: List các tuple (node_u, node_v) cần dự đoán
    :return: DataFrame chứa điểm số của các cặp sản phẩm
    """
    print(f"Đang chạy dự đoán liên kết cho {len(candidate_pairs)} cặp sản phẩm...")
    start_time = time.time()
    
    results = []
    
    # 1. Tính Jaccard Coefficient
    print("- Tính Jaccard...")
    jaccard_preds = nx.jaccard_coefficient(G, ebunch=candidate_pairs)
    jaccard_dict = {(u, v): p for u, v, p in jaccard_preds}
    
    # 2. Tính Adamic-Adar
    print("- Tính Adamic-Adar...")
    try:
        adamic_preds = nx.adamic_adar_index(G, ebunch=candidate_pairs)
        adamic_dict = {(u, v): p for u, v, p in adamic_preds}
    except nx.NetworkXError as e:
        print(f"Lỗi Adamic-Adar (có thể do node degree = 0 hoặc 1): {e}")
        adamic_dict = {(u, v): 0 for u, v in candidate_pairs}

    # Tổng hợp kết quả
    for u, v in candidate_pairs:
        results.append({
            'Product_A': u,
            'Product_B': v,
            'Jaccard_Score': jaccard_dict.get((u, v), 0),
            'Adamic_Adar_Score': adamic_dict.get((u, v), 0)
        })
        
    df_predictions = pd.DataFrame(results)
    print(f"Hoàn thành dự đoán! Thời gian: {time.time() - start_time:.2f}s")
    
    return df_predictions


def predict_links_with_explanations(
    G: nx.Graph,
    candidate_pairs,
    *,
    max_common_neighbors: int = 10,
):
    """Like `predict_links` but adds lightweight explainable signals.

    This keeps the project dependency-light (only networkx/pandas) and produces
    columns that are easy to narrate in an Explainable AI section.
    """

    try:
        # Preferred when project root is on PYTHONPATH
        from src.explainability import explain_pair
    except Exception:
        # Fallback when imported as a package module
        from .explainability import explain_pair

    df = predict_links(G, candidate_pairs)
    common_counts = []
    same_community = []
    shortest_path_len = []

    for _, row in df.iterrows():
        u = row["Product_A"]
        v = row["Product_B"]
        # We treat Adamic-Adar as a convenient default score to explain.
        exp = explain_pair(
            G,
            u,
            v,
            score=float(row.get("Adamic_Adar_Score", 0.0)),
            max_common_neighbors=max_common_neighbors,
        )
        common_counts.append(exp.signals.get("common_neighbors_count", 0))
        same_community.append(exp.signals.get("same_community", False))
        shortest_path_len.append(exp.signals.get("shortest_path_len"))

    df["Common_Neighbors_Count"] = common_counts
    df["Same_Community"] = same_community
    df["Shortest_Path_Len"] = shortest_path_len
    return df

def generate_candidate_pairs(G, num_samples=10000):
    """
    Hàm tiện ích: Sinh ra một lượng nhỏ các cặp tiềm năng (khoảng cách = 2) để test
    """
    import random
    candidates = set()
    nodes = list(G.nodes())
    
    while len(candidates) < num_samples:
        u = random.choice(nodes)
        # Lấy hàng xóm của hàng xóm
        neighbors = list(G.neighbors(u))
        if not neighbors: continue
        
        n1 = random.choice(neighbors)
        n1_neighbors = list(G.neighbors(n1))
        
        for v in n1_neighbors:
            if u != v and not G.has_edge(u, v):
                # Lưu (min, max) để tránh trùng lặp (u,v) và (v,u)
                pair = (min(u, v), max(u, v))
                candidates.add(pair)
                if len(candidates) >= num_samples:
                    break
                    
    return list(candidates)

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.join(script_dir, "..") 
    
    print("1. Đang đọc dữ liệu đồ thị...")

    file_path = os.path.join(project_root, "data", "processed", "amazon_lcc_cleaned.pkl")
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    
    print("2. Bắt đầu tính toán...")
    candidate_pairs = generate_candidate_pairs(G, num_samples=5000)
    df_result = predict_links(G, candidate_pairs)
    print("3. Đang lưu kết quả ra file CSV...")
    
    # 4. Tự động tạo thư mục lưu file 
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Lưu file
    save_path = os.path.join(output_dir, "link_prediction_test.csv")
    df_result.to_csv(save_path, index=False)
    
    print(f"🎉 Đã test xong! File được lưu an toàn tại: {save_path}")

