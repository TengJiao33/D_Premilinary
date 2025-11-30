import pandas as pd
import urllib.parse

# 1. 基础设置
base_url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

# 2. 构造查询：只取 2017-2019 (疫情前最稳定的三年)
# 依然只关注 MANHATTAN 和 Rodent
where_clause = "borough='MANHATTAN' AND created_date >= '2017-01-01' AND created_date <= '2019-12-31' AND (complaint_type like '%Rodent%')"

# 选取的列保持一致，方便后面合并
select_clause = "unique_key, created_date, complaint_type, location_type, latitude, longitude, community_board"

# 3. 拼接 URL
query_url = f"{base_url}?$where={urllib.parse.quote(where_clause)}&$select={urllib.parse.quote(select_clause)}&$limit=200000"

print("正在抓取 [2017-2019] 疫情前基准数据...")

try:
    df_old = pd.read_csv(query_url)
    df_old['created_date'] = pd.to_datetime(df_old['created_date'])

    print("-" * 30)
    print(f"疫情前数据抓取成功！")
    print(f"总行数: {len(df_old)}")
    print(f"时间范围: {df_old['created_date'].min()} 到 {df_old['created_date'].max()}")
    print("-" * 30)

    # 保存
    df_old.to_csv("Manhattan_Rodents_2017_2019_Baseline.csv", index=False)
    print("已保存为: Manhattan_Rodents_2017_2019_Baseline.csv")

except Exception as e:
    print(f"抓取失败: {e}")