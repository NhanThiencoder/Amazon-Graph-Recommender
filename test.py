import sqlite3
import pandas as pd

# Kết nối thẳng vào file database
conn = sqlite3.connect("data/data.db")

# Query tìm các Siêu Hub: Bản thân có tên VÀ sở hữu nhiều vệ tinh CÓ TÊN nhất
query = """
    SELECT p1.product_id AS Hub_ID,
           p1.title AS Ten_San_Pham,
           COUNT(r.recommended_id) as So_Luong_Ve_Tinh_Co_Ten
    FROM recommendations r
    
    -- Join lần 1: Lấy tên cho Hub
    INNER JOIN products p1 
        ON p1.product_id = CASE 
            WHEN r.product_id LIKE '%.0' THEN substr(r.product_id, 1, length(r.product_id) - 2) 
            ELSE r.product_id 
        END
        
    -- Join lần 2: Đảm bảo Vệ tinh cũng phải có tên (BÍ QUYẾT LÀ ĐÂY)
    INNER JOIN products p2
        ON p2.product_id = CASE 
            WHEN r.recommended_id LIKE '%.0' THEN substr(r.recommended_id, 1, length(r.recommended_id) - 2) 
            ELSE r.recommended_id 
        END
        
    WHERE p1.title IS NOT NULL AND p1.title != ''
      AND p2.title IS NOT NULL AND p2.title != ''
      
    GROUP BY p1.product_id, p1.title
    ORDER BY So_Luong_Ve_Tinh_Co_Ten DESC
    LIMIT 10
"""

df_hubs = pd.read_sql(query, conn)
print("=== TOP 10 SẢN PHẨM HOÀN HẢO 100% ĐỂ DEMO THUYẾT TRÌNH ===")
pd.set_option('display.max_colwidth', None)
print(df_hubs)

conn.close()