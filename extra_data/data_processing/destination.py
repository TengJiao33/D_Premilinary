import pandas as pd

# 读取你刚才抓取的那份 2023-2025 的数据
df = pd.read_csv("Manhattan_Rodents_2023_2025.csv") # 或者是你保存的那个文件名

# 检查 location_type 的分布
print("=== 老鼠出没地点分布 (Top 10) ===")
print(df['location_type'].value_counts().head(10))

# 检查一下有多少是关于 "Commercial" (商业) 或 "Street" (街道) 的
# 如果大部分是 "3+ Family Apt. Building"，我们需要思考怎么解释