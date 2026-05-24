import networkx as nx
import pandas as pd
import time
import os
import pickle

def run_louvain_clustering(G):
    """
    Chạy thuật toán Louvain để tìm các cụm cộng đồng mua sắm.
    :param G: Đồ thị NetworkX (LCC đã được tiền xử lý)
    :return: Đồ thị G đã được cập nhật thuộc tính 'community' và DataFrame kết quả
    """
    print("Đang chạy thuật toán Louvain Community Detection...")
    start_time = time.time()
    
    # Chạy Louvain
    communities = nx.community.louvain_communities(G, seed=42)
    
    # Tạo mapping để gán nhãn cho từng node
    community_map = {}
    nodes_data = []
    
    for cluster_id, comm in enumerate(communities):
        for node in comm:
            community_map[node] = cluster_id
            nodes_data.append({'ProductID': node, 'CommunityID': cluster_id})
            
    # Gán thuộc tính vào đồ thị
    nx.set_node_attributes(G, community_map, 'community')
    
    print(f"Hoàn thành! Tìm thấy {len(communities)} cộng đồng. Thời gian: {time.time() - start_time:.2f}s")
    
    df_communities = pd.DataFrame(nodes_data)
    return G, df_communities


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.join(script_dir, "..") 
    
    print("1. Đang đọc dữ liệu đồ thị...")

    file_path = os.path.join(project_root, "data", "processed", "amazon_lcc_cleaned.pkl")
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    
    print("2. Bắt đầu tính toán Louvain...")
    G, df_result = run_louvain_clustering(G)
    
    print("3. Đang lưu kết quả ra file CSV...")
    
    # 4. Tự động tạo thư mục lưu file 
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Lưu file
    save_path = os.path.join(output_dir, "louivan_test.csv")
    df_result.to_csv(save_path, index=False)
    
    print(f"🎉 Đã test xong! File được lưu an toàn tại: {save_path}")