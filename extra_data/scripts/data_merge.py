import pandas as pd

# ==========================================
# 配置：你刚下载的 ACS 文件名
# ==========================================
# 请确保这些文件已经放在你的项目文件夹里
FILE_DEMO = "extra_data/population_economy_data/Dem_1923_CDTA.xlsx"  # 人口文件
FILE_ECON = "extra_data/population_economy_data/Econ_1923_CDTA.xlsx"  # 经济文件 (根据你实际下载的名字改，可能是 Eco_xxx)

print(">>> 正在侦察 ACS 数据结构...")


def inspect_excel(filepath, label):
    try:
        print(f"\n--- 正在读取 {label} 文件: {filepath} ---")
        # ACS 文件通常前几行是元数据，真正的表头可能在第 2 行或第 3 行
        # 我们先读取前 5 行看看长什么样
        df = pd.read_excel(filepath)

        print("前 5 行预览:")
        print(df.head())

        print("\n列名列表 (寻找 'Pop' 或 'Income' 相关的关键字):")
        for col in df.columns:
            # 打印包含关键信息的列名
            if any(x in str(col).lower() for x in ['pop', 'income', 'median', 'geo', 'cdta']):
                print(f"  - {col}")

    except Exception as e:
        print(f"❌ 读取失败: {e} (请检查文件名是否正确)")


# 运行检查
inspect_excel(FILE_DEMO, "人口 (Demographic)")
inspect_excel(FILE_ECON, "经济 (Economic)")