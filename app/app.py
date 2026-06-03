import os
import sqlite3
import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# ---------------------------------------------------------
# 1. CẤU HÌNH & DATABASE
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Amazon RecSys & XAI")

# Tùy chỉnh CSS giao diện (Sáng sủa, hiện đại, khắc phục lỗi chữ chìm)
st.markdown(
    """
    <style>
    /* Nền ứng dụng chuyển sắc nhẹ nhàng */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%);
    }
    
    /* Ép màu chữ tối để hiển thị rõ trên nền sáng */
    h1, h2, h3, h4, h5, h6, p, label, div[data-testid="stMarkdownContainer"] {
        color: #2d3436 !important; 
    }
    
    /* Giữ màu chữ trắng riêng cho nút bấm */
    .stButton>button p, .stButton>button div {
        color: #ffffff !important;
    }

    /* Màu nền sidebar sáng và viền mờ */
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e1e4e8;
    }
    
    /* Tùy chỉnh nút bấm với tone màu Tím công nghệ */
    .stButton>button {
        background-color: #6c5ce7 !important; 
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5849c4 !important;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3);
    }
    
    /* Bo góc và làm đậm chữ cho các thẻ thông báo (st.info, st.success) */
    .stAlert {
        border-radius: 12px;
    }
    .stAlert p {
        color: #1e272e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Đường dẫn database
current_dir = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.abspath(
    os.path.join(current_dir, '..', 'data', 'data.db')
)

# ---------------------------------------------------------
# 2. HÀM ĐỌC DỮ LIỆU
# ---------------------------------------------------------
@st.cache_data
def get_all_products():
    """
    Lấy danh sách sản phẩm có recommendation
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        query = """
            SELECT DISTINCT p.product_id, p.title
            FROM products p
            INNER JOIN recommendations r
                ON p.product_id = CASE
                    WHEN r.product_id LIKE '%.0' THEN substr(r.product_id, 1, length(r.product_id) - 2)
                    ELSE r.product_id
                END
        """

        df = pd.read_sql(query, conn)
        conn.close()

        # Xử lý title null
        df['title'] = df['title'].fillna("Sản phẩm Amazon")

        return dict(zip(df['product_id'], df['title']))

    except Exception as e:
        st.error(f"Lỗi đọc bảng products: {e}")
        return {}


@st.cache_data
def get_recommendations(product_id, limit):
    """Lấy danh sách gợi ý từ bảng recommendations với số lượng tùy chỉnh"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"""
            SELECT CASE
                    WHEN recommended_id LIKE '%.0' THEN substr(recommended_id, 1, length(recommended_id) - 2)
                    ELSE recommended_id
                END AS target,
                score,
                'Dựa trên cấu trúc mạng lưới' AS reason
            FROM recommendations
            WHERE CASE
                    WHEN product_id LIKE '%.0' THEN substr(product_id, 1, length(product_id) - 2)
                    ELSE product_id
                END = '{product_id}'
            ORDER BY score DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Lỗi đọc bảng recommendations: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------
# 3. GIAO DIỆN CHÍNH
# ---------------------------------------------------------
st.title("🛒 Hệ thống Gợi ý Sản phẩm (Explainable AI)")
st.markdown("---")

product_dict = get_all_products()

# ---------------------------------------------------------
# KIỂM TRA DỮ LIỆU
# ---------------------------------------------------------
if not product_dict:
    st.warning("Không tìm thấy dữ liệu sản phẩm. Vui lòng kiểm tra lại file data.db.")

else:

    st.sidebar.title("🏷️ Chọn sản phẩm")
    st.sidebar.markdown(
        "Chọn một sản phẩm để khám phá gợi ý liên kết và đồ thị tương tác."
    )

    # Sidebar chọn sản phẩm
    selected_product_id = st.sidebar.selectbox(
        "Chọn sản phẩm khách hàng đang xem:",
        options=list(product_dict.keys()),
        format_func=lambda x: f"{x} - {product_dict[x]}"
    )
    
    # THÊM THANH KÉO: Cho phép chọn từ 1 đến 20 gợi ý
    num_recs = st.sidebar.slider(
        "Số lượng gợi ý hiển thị trên đồ thị:",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    selected_product_name = product_dict.get(
        selected_product_id,
        selected_product_id
    )

    # Lấy recommendation với limit truyền vào
    df_recs = get_recommendations(selected_product_id, limit=num_recs)

    # Layout - Tăng không gian cho cột trái để chữ không bị rớt dòng
    col1, col2 = st.columns([1.5, 2.2]) 

    # ---------------------------------------------------------
    # CỘT TRÁI
    # ---------------------------------------------------------
    with col1:

        st.markdown("### 📦 Sản phẩm đang xem")

        st.info(
            f"""
            **{selected_product_name}**

            ID: {selected_product_id}
            """
        )

        st.markdown("### 💡 Có thể bạn cũng thích")

        if not df_recs.empty:
            
           # Tạo một khu vực có thanh cuộn (scroll)
            with st.container(height=480, border=False):
                for _, row in df_recs.iterrows():

                    rec_id = str(row['target']) # Đảm bảo ép về kiểu chuỗi để tìm kiếm chuẩn xác
                    score = row['score']

                    # Lấy tên, nếu không có thì gán text mặc định
                    rec_name = product_dict.get(rec_id, "Sản phẩm chưa cập nhật tên")

                    st.success(
                        f"""
                        **{rec_name}** *(ID: {rec_id})*

                        Độ tương đồng: **{score:.4f}**
                        """
                    )

        else:
            st.warning("Chưa có recommendation cho sản phẩm này.")

    # ---------------------------------------------------------
    # CỘT PHẢI - ĐỒ THỊ XAI
    # ---------------------------------------------------------
    with col2:

        st.markdown("### 🕸️ Đồ thị giải thích (Explainable AI)")

        st.markdown(
            "Kéo thả hoặc di chuột lên sản phẩm để khám phá mạng lưới các sản phẩm liên quan trực tiếp."
        )

        if not df_recs.empty:

            # Tạo graph
            G = nx.Graph()

            # Node nguồn
            G.add_node(
                selected_product_id,
                label=selected_product_name,
                group="Source"
            )

            # Recommendation nodes
            for _, row in df_recs.iterrows():

                target_id = row['target']
                score = row['score']

                target_name = product_dict.get(target_id, target_id)

                G.add_node(
                    target_id,
                    label=target_name,
                    group="Recommendation"
                )

                G.add_edge(
                    selected_product_id,
                    target_id,
                    title=f"Score: {score:.4f}"
                )

            # Pyvis network
            net = Network(
                height="700px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black",
                directed=False,
                notebook=False,
            )

            net.barnes_hut(
                gravity=-22000,
                central_gravity=0.1,
                spring_length=140,
                spring_strength=0.08,
                damping=0.35,
            )

            # Add nodes với màu sắc mới trực quan hơn
            for node, data in G.nodes(data=True):

                if node == selected_product_id:
                    color = "#ff7675"  # Đỏ san hô cho sản phẩm đang xem
                    size = 44
                else:
                    color = "#0984e3"  # Xanh dương cho sản phẩm gợi ý
                    size = 26

                net.add_node(
                    node,
                    label=data['label'],
                    title=f"{data['label']}\nID: {node}",
                    color=color,
                    size=size,
                    shape='dot',
                )

            # Add edges
            for source, target, data in G.edges(data=True):

                net.add_edge(
                    source,
                    target,
                    title=data['title'],
                    label=data['title'],
                    color='#b2bec3',  # Xám nhạt cho đường nối giúp UI thanh thoát hơn
                    width=2,
                    smooth={'type': 'dynamic'},
                )

            net.set_options(
                """
                var options = {
                    "interaction": {
                        "hover": true,
                        "hoverConnectedEdges": true,
                        "selectConnectedEdges": true,
                        "tooltipDelay": 100,
                        "zoomView": true,
                        "dragView": true
                    },
                    "physics": {
                        "enabled": true,
                        "barnesHut": {
                            "gravitationalConstant": -22000,
                            "centralGravity": 0.15,
                            "springLength": 130,
                            "springConstant": 0.08,
                            "damping": 0.35
                        }
                    },
                    "nodes": {
                        "borderWidth": 2,
                        "font": {
                            "size": 16,
                            "face": "Segoe UI"
                        }
                    },
                    "edges": {
                        "color": {"inherit": true},
                        "smooth": {"type": "dynamic"}
                    }
                }
                """
            )

            # Hiển thị HTML
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:

                net.save_graph(tmp_file.name)

                with open(tmp_file.name, 'r', encoding='utf-8') as HtmlFile:

                    components.html(
                        HtmlFile.read(),
                        height=600
                    )

        else:
            st.info("Không có dữ liệu để vẽ đồ thị.")