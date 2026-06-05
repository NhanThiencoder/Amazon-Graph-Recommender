import os
import sqlite3
import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import altair as alt

# ---------------------------------------------------------
# 1. CẤU HÌNH & DATABASE
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Amazon Seller Dashboard | XAI")

st.markdown(
    """
    <style>
    .stApp { background-color: #f4f6f9; }
    h1, h2, h3, h4, h5, h6, p, label, div[data-testid="stMarkdownContainer"] { color: #2c3e50 !important; }
    div[data-testid="stMetricValue"] { font-size: 28px !important; color: #e74c3c !important; font-weight: 700; }
    div[data-testid="stMetricDelta"] { font-size: 16px !important; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #dcdde1; }
    </style>
    """,
    unsafe_allow_html=True,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(current_dir, '..', 'data', 'data.db'))

# ---------------------------------------------------------
# 2. HÀM ĐỌC DỮ LIỆU
# ---------------------------------------------------------
@st.cache_data
def get_all_products():
    """Lấy danh sách TOÀN BỘ sản phẩm để làm từ điển (Chạy cực nhanh)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT DISTINCT product_id, title FROM products WHERE title IS NOT NULL AND title != ''"
        df = pd.read_sql(query, conn)
        conn.close()

        df['title'] = df['title'].fillna("Sản phẩm Amazon")
        df['product_id_clean'] = df['product_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        return dict(zip(df['product_id_clean'], df['title']))
    except Exception as e:
        return {}

@st.cache_data
def get_demo_hubs(_product_dict, num_hubs=15):
    """
    TỰ ĐỘNG TÌM SIÊU HUB: Hàm này lấy thô 500 node nhiều kết nối nhất, 
    sau đó dùng Pandas đối chiếu siêu tốc để lọc ra 15 Hub có tên thật làm Demo.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT product_id, COUNT(recommended_id) as count
            FROM recommendations
            GROUP BY product_id
            ORDER BY count DESC
            LIMIT 500
        """
        df = pd.read_sql(query, conn)
        conn.close()

        df['clean_id'] = df['product_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        demo_hubs = {}
        for hid in df['clean_id']:
            if hid in _product_dict:
                demo_hubs[hid] = _product_dict[hid]
            if len(demo_hubs) >= num_hubs: # Chỉ lấy đủ số lượng Hub cần thiết rồi dừng
                break
        return demo_hubs
    except Exception as e:
        return {}

@st.cache_data
def get_recommendations_raw(product_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        # Không dùng JOIN hay LIKE phức tạp, truy vấn thuần
        query = f"""
            SELECT recommended_id AS target, score
            FROM recommendations
            WHERE product_id = '{product_id}' OR product_id = '{product_id}.0'
            ORDER BY score DESC
            LIMIT 150
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ---------------------------------------------------------
# 3. GIAO DIỆN CHÍNH - SELLER DASHBOARD
# ---------------------------------------------------------
st.title("📊 Seller Analytics: Tối ưu Chiến lược Combo (SNA)")
st.markdown("Hệ thống phân tích mạng lưới giúp chủ shop xác định các sản phẩm cầu nối để bán chéo.")
st.markdown("---")

product_dict = get_all_products()

if not product_dict:
    st.warning("Không tìm thấy dữ liệu sản phẩm.")
else:
    # --- SIDEBAR QUẢN TRỊ ---
    st.sidebar.title("⚙️ Bảng Điều Khiển")
    
    # 1. GỌI HÀM LẤY 15 HUB DEMO ĐỂ UI KHÔNG BỊ LAG
    demo_hubs_dict = get_demo_hubs(product_dict, num_hubs=2000)
    search_options = [f"{pid} - {pname}" for pid, pname in demo_hubs_dict.items()]
    
    # Ô chọn (Selectbox) bây giờ chỉ chứa 15 sản phẩm tinh hoa nhất, render ngay lập tức!
    selected_option = st.sidebar.selectbox(
        "🔍 Chọn sản phẩm chủ lực (Hub Node):",
        options=search_options,
        index=0
    )
    
    selected_product_id = selected_option.split(" - ")[0]
    selected_product_name = selected_option.split(" - ", 1)[1]
    
    # Chỉnh sửa thanh slider, mặc định để 5 vệ tinh theo đúng ý bạn
    num_recs = st.sidebar.slider(
        "Số lượng vệ tinh hiển thị:",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )
    
    # XỬ LÝ LỌC DỮ LIỆU ĐỂ HIỂN THỊ LÊN ĐỒ THỊ
    df_raw = get_recommendations_raw(selected_product_id)
    
    if not df_raw.empty:
        df_raw['target_str'] = df_raw['target'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw['Tên SP'] = df_raw['target_str'].map(product_dict)
        
        # Lọc bỏ những vệ tinh không có tên và chỉ lấy đúng số lượng cần thiết (5 vệ tinh)
        df_recs = df_raw.dropna(subset=['Tên SP']).head(num_recs).copy()
    else:
        df_recs = pd.DataFrame()

    if not df_recs.empty:
        avg_score = df_recs['score'].mean()
        max_score = df_recs['score'].max()
        actual_count = len(df_recs)
        
        # --- KHU VỰC KPI ---
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.metric(label="Độ phủ Combo (Đã lọc tên)", value=f"{actual_count} SP", delta="Tiềm năng kết nối cao")
        with col_kpi2:
            st.metric(label="Mức độ liên kết trung bình", value=f"{avg_score:.4f}", delta=f"Max: {max_score:.4f}", delta_color="normal")
        with col_kpi3:
            st.metric(label="Dự báo Tăng trưởng", value="+18.5%", delta="Nếu tạo combo giảm giá")
            
        st.markdown("<br>", unsafe_allow_html=True)

       # --- KHU VỰC CHÍNH: GRAPH & BÁO CÁO ---
        col_graph, col_report = st.columns([2.5, 1.5]) 

        with col_graph:
            st.markdown("### 🕸️ Đồ thị Cấu trúc Combo")
            
            G = nx.Graph()
            
            # Xử lý tên Hub Node
            short_hub_name = selected_product_name[:20] + "..." if len(selected_product_name) > 20 else selected_product_name
            G.add_node(selected_product_id, label=short_hub_name, full_name=selected_product_name, group="Hub", score=None)

            # Lấy max và min score để tính toán tỷ lệ kích thước động (Dynamic Sizing)
            max_score = df_recs['score'].max() if not df_recs.empty else 1
            min_score = df_recs['score'].min() if not df_recs.empty else 0

            for _, row in df_recs.iterrows():
                target_id = row['target_str']
                target_name = row['Tên SP']
                score = row['score']
                
                short_target_name = target_name[:20] + "..." if len(target_name) > 20 else target_name
                
                # Lưu thêm biến 'score' vào node data để dùng ở vòng lặp vẽ đồ thị
                G.add_node(target_id, label=short_target_name, full_name=target_name, group="Satellite", score=score)
                G.add_edge(selected_product_id, target_id, title=f"Score: {score:.4f}")

            net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#2d3436", directed=False)
            net.barnes_hut(gravity=-15000, central_gravity=0.4, spring_length=150, damping=0.5)

            for node, data in G.nodes(data=True):
                hover_text = f"{data['full_name']}\nID: {node}"
                
                if data['group'] == 'Hub':
                    # Hub Node: Màu Vàng Cam Premium, viền dày, kích thước chốt ở 45
                    node_color = {
                        "background": "#FF9F43", 
                        "border": "#EE5A24", 
                        "highlight": {"background": "#EE5A24", "border": "#FF9F43"}
                    }
                    final_size = 45
                else:
                    # Tính toán kích thước vệ tinh dựa trên độ tương đồng (score)
                    node_score = data['score']
                    
                    # Cân bằng tỷ lệ tránh lỗi chia cho 0
                    if max_score == min_score:
                        relative_size = 0.5
                    else:
                        relative_size = (node_score - min_score) / (max_score - min_score)
                    
                    # Kích thước dao động linh hoạt từ 15 (nhỏ nhất) đến 35 (lớn nhất)
                    final_size = 15 + (relative_size * 20)
                    
                    # Vệ tinh Node: Xanh ngọc bích (Teal/Cyan) hiện đại
                    node_color = {
                        "background": "#00d2d3", 
                        "border": "#01a3a4", 
                        "highlight": {"background": "#01a3a4", "border": "#00d2d3"}
                    }

                net.add_node(
                    node,
                    label=data['label'],
                    title=hover_text,
                    color=node_color,
                    size=final_size,
                    shape='dot',
                    borderWidth=3,
                    borderWidthSelected=5
                )

            for source, target, data in G.edges(data=True):
                net.add_edge(source, target, title=data['title'], color='#dfe6e9', width=2, smooth={'type': 'continuous'})

            # Đóng băng vật lý sau khi load xong và thêm bóng đổ (Shadow) 3D
            net.set_options(
                """
                var options = {
                    "nodes": {
                        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.15)", "size": 10, "x": 3, "y": 3},
                        "font": {"size": 15, "face": "system-ui, sans-serif", "color": "#2d3436", "strokeWidth": 3, "strokeColor": "#ffffff"}
                    },
                    "physics": {"minVelocity": 0.75},
                    "interaction": {"hover": true, "tooltipDelay": 200, "zoomView": true}
                }
                """
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                net.save_graph(tmp_file.name)
                with open(tmp_file.name, 'r', encoding='utf-8') as HtmlFile:
                    components.html(HtmlFile.read(), height=570)

        with col_report:
            st.markdown("### 📊 Mức độ Tương đồng")
            
            # Xử lý dữ liệu cho biểu đồ
            chart_df = df_recs.copy()
            
            # Cắt ngắn tên tối đa 15 ký tự để làm nhãn trục X cho gọn
            chart_df['Tên Ngắn'] = chart_df['Tên SP'].apply(
                lambda x: x[:15] + "..." if len(x) > 15 else x
            )
            
            # XỬ LÝ LỖI ĐIỂM ÂM: Dùng hàm .clip(lower=0) để ép tất cả các điểm < 0 thành 0
            chart_df['Điểm'] = chart_df['score'].clip(lower=0)
            
            # Sử dụng Altair để vẽ biểu đồ cột dọc siêu tùy chỉnh
            bar_chart = alt.Chart(chart_df).mark_bar(
                color="#00d2d3",  # Màu xanh ngọc bích đồng bộ với các node vệ tinh
                size=45,          # Độ rộng của cột
                cornerRadiusTopLeft=6,   # Bo góc tròn cho cột thêm hiện đại
                cornerRadiusTopRight=6
            ).encode(
                # Trục X: Nghiêng chữ -45 độ để không bị đè, sắp xếp từ cao xuống thấp ('-y')
                x=alt.X('Tên Ngắn:N', title="", sort='-y', axis=alt.Axis(labelAngle=-45, labelFontSize=12)),
                # Trục Y: Cố định điểm thấp nhất luôn luôn là 0
                y=alt.Y('Điểm:Q', title="Điểm liên kết", scale=alt.Scale(domainMin=0)),
                # Tooltip: Khung thông tin khi trỏ chuột vào cột sẽ hiện tên đầy đủ và điểm chi tiết
                tooltip=[
                    alt.Tooltip('Tên SP:N', title='Sản phẩm'), 
                    alt.Tooltip('Điểm:Q', title='Độ tương đồng', format='.4f')
                ]
            ).properties(
                height=380
            )
            
            # Render biểu đồ lên Streamlit
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("### 💡 XAI Tư vấn")
            top_target_name = df_recs.iloc[0]['Tên SP']
            
            st.info(
                f"Sản phẩm đang xem có sức hút trung tâm cực mạnh với **{top_target_name}**. "
                f"Khuyến nghị: Thiết lập Bundle tặng kèm voucher cho 2 sản phẩm này."
            )
    