import pandas as pd
import urllib.parse

# DSNY Monthly Tonnage Data
# 这个数据集记录了每个区、每个月收集了多少吨垃圾
base_url = "https://data.cityofnewyork.us/resource/ebb7-mvp5.csv"

# 逻辑：
# 1. 只要 MANHATTAN
# 2. 时间要跟你的老鼠数据对齐 (2017年至今，覆盖所有你的分析时段)
where_clause = "borough='Manhattan' AND month >= '2017 / 01'"

# limit 设大一点，虽然只有几百行，但为了保险
query_url = f"{base_url}?$where={urllib.parse.quote(where_clause)}&$limit=50000"

print("正在抓取曼哈顿垃圾吨数数据 (Tonnage)...")

try:
    df_trash = pd.read_csv(query_url)

    # 简单的列名清洗（API返回的列名有时候很乱）
    print("列名预览:", df_trash.columns)

    # 保存
    df_trash.to_csv("Manhattan_Garbage_Tonnage.csv", index=False)
    print(f"成功！共抓取 {len(df_trash)} 行垃圾产量数据。")

except Exception as e:
    print(f"抓取失败: {e}")