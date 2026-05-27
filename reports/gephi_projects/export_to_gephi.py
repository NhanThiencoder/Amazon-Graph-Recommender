import sys
import io
import os
import pickle
import pandas as pd
import networkx as nx
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.join(script_dir, "..") 
processed_dir = os.path.join(project_root, "data", "processed")
gephi_dir = os.path.join(project_root, "reports", "gephi_projects")

os.makedirs(gephi_dir, exist_ok=True)

print("1. Đang tải đồ thị và thuộc tính...")
with open(os.path.join(processed_dir, 'amazon_lcc_cleaned.pkl'), 'rb') as f:
    G = pickle.load(f)

df_louvain = pd.read_csv(os.path.join(processed_dir, 'louivan_test.csv'))
df_centrality = pd.read_csv(os.path.join(processed_dir, 'centrality_test.csv'))

community_dict = {int(row['ProductID']): int(row['CommunityID']) for _, row in df_louvain.iterrows()}
degree_dict = {int(row['ProductID']): float(row['Degree_Centrality']) for _, row in df_centrality.iterrows()}

print("2. Đang tạo đồ thị dành riêng cho Gephi...")
G_clean = nx.Graph()

# Đếm và lọc Top 10 cộng đồng
community_counts = Counter(community_dict.values())
top_10_communities = [cmty_id for cmty_id, count in community_counts.most_common(10)]

print("3. Bắt đầu nhúng node và edge hợp lệ...")
for node in G.nodes():
    cmty = community_dict.get(node)
    if cmty in top_10_communities:
        deg = degree_dict.get(node, 0.0)
        G_clean.add_node(int(node), CommunityID=int(cmty), Degree_Centrality=float(deg))

for u, v in G.edges():
    if G_clean.has_node(u) and G_clean.has_node(v):
        G_clean.add_edge(int(u), int(v))

print(f"-> Quy mô đồ thị chuẩn: {G_clean.number_of_nodes():,} nodes và {G_clean.number_of_edges():,} edges.")

print("4. Xuất file định dạng .gexf an toàn...")
output_path = os.path.join(gephi_dir, 'amazon_top_communities.gexf')

nx.write_gexf(G_clean, output_path)

print(f"-> Hoàn tất! Hãy mở file mới '{output_path}' bằng Gephi.")