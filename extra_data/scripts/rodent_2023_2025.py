import pandas as pd
import urllib.parse

# 1. 设置基础 URL
base_url = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

# 2. 构造查询条件 (SoQL)
# 策略：
# (1) borough='MANHATTAN': 锁定题目区域
# (2) created_date > '2021-01-01': 获取最近3-4年的数据，包含完整的季节性周期
# (3) complaint_type: 模糊匹配 'Rodent' (包含 Rat Sighting, Mouse Sighting, Condition Attracting Rodents 等)
where_clause = "borough='MANHATTAN' AND created_date > '2023-01-01' AND complaint_type like '%Rodent%'"

# 选取的列：
# unique_key: 唯一ID，防止重复
# created_date: 时间序列分析用
# complaint_type: 具体是看见老鼠还是看见老鼠洞
# location_type: 是在住宅楼(3+ Family Apt)还是商业区，对分析很重要
# latitude, longitude: 画地图用
# community_board: 用来和你的 DSNY 地图分区匹配 (关键!)
select_clause = "unique_key, created_date, complaint_type, location_type, latitude, longitude, community_board"

# 3. 拼接 URL (设置 limit 为 20万，管够)
query_url = f"{base_url}?$where={urllib.parse.quote(where_clause)}&$select={urllib.parse.quote(select_clause)}&$limit=200000"

print("正在通过 API 抓取曼哈顿 2023年至今的老鼠数据...")

try:
    # 读取数据
    df_rats = pd.read_csv(query_url)

    # 转换日期格式
    df_rats['created_date'] = pd.to_datetime(df_rats['created_date'])

    print("-" * 30)
    print(f"数据抓取成功！")
    print(f"总行数: {len(df_rats)}")
    print(f"时间跨度: {df_rats['created_date'].min()} 到 {df_rats['created_date'].max()}")
    print("-" * 30)
    print("前5行预览:")
    print(df_rats.head())

    # 保存
    filename = "../rodent_data/Manhattan_Rodents_2023_2025.csv"
    df_rats.to_csv(filename, index=False)
    print(f"文件已保存为: {filename}")

except Exception as e:
    print(f"抓取失败，原因: {e}")