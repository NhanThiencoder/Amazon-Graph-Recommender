import networkx as nx
import pandas as pd
import time
import pickle
import os

def calculate_centrality_metrics(G):
    """
    Tính toán Degree Centrality và PageRank cho các sản phẩm.
    :param G: Đồ thị NetworkX
    :return: Đồ thị G đã cập nhật thuộc tính và DataFrame chứa kết quả
    """
    print("Đang tính toán Degree Centrality...")
    start_time = time.time()
    degree_cent = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_cent, 'degree_centrality')
    print(f"Xong Degree Centrality. Thời gian: {time.time() - start_time:.2f}s")
    
    print("Đang tính toán PageRank...")
    start_time = time.time()
    # Điều chỉnh alpha = 0.85 (tiêu chuẩn)
    pagerank_cent = nx.pagerank(G, alpha=0.85)
    nx.set_node_attributes(G, pagerank_cent, 'pagerank')
    print(f"Xong PageRank. Thời gian: {time.time() - start_time:.2f}s")
    
    # Tổng hợp dữ liệu
    nodes_data = []
    for node in G.nodes():
        node_info = {
            'ProductID': node,
            'Degree_Centrality': G.nodes[node].get('degree_centrality', 0),
            'PageRank': G.nodes[node].get('pagerank', 0)
        }
        # Nếu đã chạy community trước đó, lấy luôn
        if 'community' in G.nodes[node]:
            node_info['CommunityID'] = G.nodes[node]['community']
            
        nodes_data.append(node_info)
        
    df_centrality = pd.DataFrame(nodes_data)
    return G, df_centrality

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.join(script_dir, "..") 
    
    print("1. Đang đọc dữ liệu đồ thị...")

    file_path = os.path.join(project_root, "data", "processed", "amazon_lcc_cleaned.pkl")
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    
    print("2. Bắt đầu tính toán Centrality...")
    G, df_result = calculate_centrality_metrics(G)
    
    print("3. Đang lưu kết quả ra file CSV...")
    
    # 4. Tự động tạo thư mục lưu file 
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Lưu file
    save_path = os.path.join(output_dir, "centrality_test.csv")
    df_result.to_csv(save_path, index=False)
    
    print(f"🎉 Đã test xong! File được lưu an toàn tại: {save_path}")