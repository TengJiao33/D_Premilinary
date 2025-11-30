import pandas as pd
import re
import os

# ==========================================
# 1. 文件路径配置 (请确保这些文件都在!)
# ==========================================
FILE_GEO = "raw_data/DSNY_Districts_20251130.csv"
FILE_TRASH_ALL = "extra_data/garbage_data/Manhattan_Garbage_Ton_201701_202510.csv"  # 包含 2017-2025 的总表
FILE_RATS_OLD = "extra_data/rodent_data/Manhattan_Rodents_2017_2019_Baseline.csv"  # 之前抓取的旧老鼠数据
# ACS 数据 (沿用 1923 数据)
FILE_DEMO = "extra_data/population_economy_data/Dem_1923_CDTA.xlsx"
FILE_ECON = "extra_data/population_economy_data/Econ_1923_CDTA.xlsx"
FILE_HOUS = "extra_data/population_economy_data/Hous_1923_CDTA.xlsx"


# ==========================================
# 2. 核心工具函数
# ==========================================
def parse_id(val):
    s = str(val).upper().strip()
    nums = re.findall(r'\d+', s)
    if not nums: return None
    num = int(nums[-1])
    if 101 <= num <= 112: return num
    if 1 <= num <= 12: return 100 + num
    return None


def process_acs(filepath, val_col, rename_col):
    """读取 ACS 数据并提取指定列"""
    if not os.path.exists(filepath):
        print(f"❌ 找不到文件: {filepath}")
        return None
    df = pd.read_excel(filepath, engine='openpyxl')
    # 找 GeoID
    geo_col = [c for c in df.columns if 'geo' in str(c).lower() and 'id' in str(c).lower()][0]
    df = df[df[geo_col].astype(str).str.startswith('MN')].copy()
    df['CD_ID'] = df[geo_col].apply(parse_id)

    # 找目标列
    target_col = None
    for c in df.columns:
        if c.lower() == val_col.lower(): target_col = c; break
    # 特殊处理经济数据的列名
    if not target_col and val_col == 'MdHHIncE':
        for c in df.columns:
            if 'med' in c.lower() and 'inc' in c.lower() and 'moe' not in c.lower():
                target_col = c;
                break

    if target_col:
        return df[['CD_ID', target_col]].rename(columns={target_col: rename_col})
    return None


# ==========================================
# 3. 开始聚合 Baseline (2017-2019)
# ==========================================
print("⏳ 正在构建 [2017-2019 基准数据集]...")

# --- A. 地理 ---
df_geo = pd.read_csv(FILE_GEO)
df_geo['CD_ID'] = df_geo['DISTRICTCODE'].apply(parse_id)
df_geo = df_geo[(df_geo['CD_ID'] >= 101) & (df_geo['CD_ID'] <= 112)][['CD_ID', 'DISTRICT', 'SHAPE_Area']].copy()

# --- B. 老鼠 (2017-2019) ---
if os.path.exists(FILE_RATS_OLD):
    print("   读取 2017-2019 老鼠数据...")
    df_rats = pd.read_csv(FILE_RATS_OLD)
    df_rats['CD_ID'] = df_rats['community_board'].apply(parse_id)
    rat_stats = df_rats.groupby('CD_ID').size().reset_index(name='Rat_Complaints')
else:
    print(f"❌ 严重错误: 找不到 {FILE_RATS_OLD}，请确认你是否运行了之前的抓取脚本。")
    rat_stats = None

# --- C. 垃圾 (从总表中切分 2017-2019) ---
print("   切分 2017-2019 垃圾数据...")
df_trash = pd.read_csv(FILE_TRASH_ALL)
df_trash['date'] = pd.to_datetime(df_trash['month'], format='%Y / %m', errors='coerce')
# 【关键】时间筛选
mask = (df_trash['date'] >= '2017-01-01') & (df_trash['date'] <= '2019-12-31')
df_trash_base = df_trash[mask].copy()

df_trash_base['CD_ID'] = df_trash_base['communitydistrict'].apply(parse_id)
df_trash_base = df_trash_base.dropna(subset=['CD_ID'])
df_trash_base['Total_Tons'] = df_trash_base['refusetonscollected'].fillna(0) + \
                              df_trash_base['papertonscollected'].fillna(0) + \
                              df_trash_base['mgptonscollected'].fillna(0)
trash_stats = df_trash_base.groupby('CD_ID')['Total_Tons'].mean().reset_index(name='Monthly_Trash_Tons')

# --- D. ACS 数据 (复用 2023 数据作为常量) ---
print("   加载 ACS 2023 数据 (作为人口基底)...")
df_pop = process_acs(FILE_DEMO, 'Pop_1E', 'Population')
df_econ = process_acs(FILE_ECON, 'MdHHIncE', 'Median_Income')
df_hous = process_acs(FILE_HOUS, 'HUs_1E', 'Housing_Units')

# --- E. 合并 ---
master = df_geo
for df in [rat_stats, trash_stats, df_pop, df_econ, df_hous]:
    if df is not None:
        master = master.merge(df, on='CD_ID', how='left')

master = master.fillna(0)

# --- F. 计算密度指标 (保持与 2023-2025 一致) ---
if 'Population' in master.columns and 'Housing_Units' in master.columns:
    # 人均垃圾 (2017-2019水平)
    master['Trash_Per_Capita'] = master.apply(
        lambda x: x['Monthly_Trash_Tons'] / x['Population'] if x['Population'] > 0 else 0, axis=1)
    # 住房老鼠密度 (2017-2019水平)
    master['Rat_Density_Per_Unit'] = master.apply(
        lambda x: x['Rat_Complaints'] / x['Housing_Units'] if x['Housing_Units'] > 0 else 0, axis=1)

# 保存
output_file = "Manhattan_Data_Baseline_2017_2019.csv"
master.to_csv(output_file, index=False)

print("-" * 30)
print(f"✅ 基准表已生成: {output_file}")
print(f"   垃圾数据行数: {len(trash_stats)}")
print(f"   老鼠数据行数: {len(rat_stats)}")
print("   人口/经济数据已成功对齐。")
print("-" * 30)