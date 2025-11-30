import pandas as pd
import re
import os

# ==========================================
# 1. 文件配置 (请根据你的实际文件名修改!!)
# ==========================================
FILE_GEO = "raw_data/DSNY_Districts_20251130.csv"
FILE_TRASH_ALL = "extra_data/garbage_data/Manhattan_Garbage_Ton_201701_202510.csv"  # 那个 2017-2025 的大文件
FILE_RATS_OLD = "extra_data/rodent_data/Manhattan_Rodents_2017_2019_Baseline.csv"  # 你的老数据
FILE_RATS_NEW = "extra_data/rodent_data/Manhattan_Rodents_2023_2025.csv"  # 你的新数据

# ACS 数据 (主要用于 Current 阶段)
FILE_DEMO = "extra_data/population_economy_data/Dem_1923_CDTA.xlsx"
FILE_ECON = "extra_data/population_economy_data/Econ_1923_CDTA.xlsx"
FILE_HOUS = "extra_data/population_economy_data/Hous_1923_CDTA.xlsx"


# ==========================================
# 2. 辅助工具
# ==========================================
def parse_id(val):
    """统一 ID 为 101-112"""
    s = str(val).upper().strip()
    nums = re.findall(r'\d+', s)
    if not nums: return None
    num = int(nums[-1])
    if 101 <= num <= 112: return num
    if 1 <= num <= 12: return 100 + num
    return None


def load_geo():
    print("🗺️  加载地图基底...")
    df = pd.read_csv(FILE_GEO)
    df['CD_ID'] = df['DISTRICTCODE'].apply(parse_id)
    return df[(df['CD_ID'] >= 101) & (df['CD_ID'] <= 112)][['CD_ID', 'DISTRICT', 'SHAPE_Area']]


def load_acs_features():
    """一次性加载所有 ACS 特征 (人口/经济/住房)"""
    print("📊 加载 ACS 2019-2023 特征 (作为当前状态)...")

    # 内部小函数：读取单文件
    def _read_acs(fp, target_col_code, rename_to):
        if not os.path.exists(fp): return None
        df = pd.read_excel(fp, engine='openpyxl')
        # 找 GeoID 列
        geo_col = [c for c in df.columns if 'geo' in str(c).lower() and 'id' in str(c).lower()][0]
        # 筛选曼哈顿
        df = df[df[geo_col].astype(str).str.startswith('MN')].copy()
        df['CD_ID'] = df[geo_col].apply(parse_id)

        # 找目标列 (模糊匹配)
        real_col = None
        for c in df.columns:
            if c.lower() == target_col_code.lower():
                real_col = c;
                break

        # 经济数据的特殊处理 (有时叫 MedInc, 有时叫 MdHHIncE)
        if not real_col and target_col_code == 'MdHHIncE':
            for c in df.columns:
                if 'med' in c.lower() and 'inc' in c.lower() and 'moe' not in c.lower():
                    real_col = c;
                    break

        if real_col:
            return df[['CD_ID', real_col]].rename(columns={real_col: rename_to})
        return None

    df_pop = _read_acs(FILE_DEMO, 'Pop_1E', 'Population')
    df_econ = _read_acs(FILE_ECON, 'MdHHIncE', 'Median_Income')
    df_hous = _read_acs(FILE_HOUS, 'HUs_1E', 'Housing_Units')

    # 合并这三个
    master_acs = df_pop
    if df_econ is not None: master_acs = master_acs.merge(df_econ, on='CD_ID', how='left')
    if df_hous is not None: master_acs = master_acs.merge(df_hous, on='CD_ID', how='left')

    return master_acs


# ==========================================
# 3. 核心：构建特定时间段的数据集
# ==========================================
def build_period_dataset(period_name, rat_file, start_date, end_date, df_geo, df_trash_all, df_acs=None):
    print(f"\n🏗️  正在构建 [{period_name}] 数据集 ({start_date} ~ {end_date})...")

    # 1. 处理老鼠 (直接读取对应时段的文件)
    if os.path.exists(rat_file):
        df_rats = pd.read_csv(rat_file)
        df_rats['CD_ID'] = df_rats['community_board'].apply(parse_id)
        rat_stats = df_rats.groupby('CD_ID').size().reset_index(name='Rat_Complaints')
    else:
        print(f"   ❌ 找不到老鼠文件: {rat_file}")
        return

    # 2. 处理垃圾 (从总表中切分时间)
    # 确保日期格式
    df_trash_all['date_obj'] = pd.to_datetime(df_trash_all['month'], format='%Y / %m', errors='coerce')

    # 切片
    mask = (df_trash_all['date_obj'] >= start_date) & (df_trash_all['date_obj'] <= end_date)
    df_trash_period = df_trash_all[mask].copy()

    # 计算吨数
    df_trash_period['Total_Tons'] = df_trash_period['refusetonscollected'].fillna(0) + \
                                    df_trash_period['papertonscollected'].fillna(0) + \
                                    df_trash_period['mgptonscollected'].fillna(0)

    df_trash_period['CD_ID'] = df_trash_period['communitydistrict'].apply(parse_id)
    trash_stats = df_trash_period.groupby('CD_ID')['Total_Tons'].mean().reset_index(name='Monthly_Trash_Tons')

    print(f"   - 老鼠数据行数 (聚合后): {len(rat_stats)}")
    print(f"   - 垃圾数据涵盖月份数: {df_trash_period['month'].nunique()}")

    # 3. 合并
    master = df_geo.merge(rat_stats, on='CD_ID', how='left')
    master = master.merge(trash_stats, on='CD_ID', how='left')

    # 4. 如果有 ACS 数据 (通常只给 Current 阶段用)
    if df_acs is not None:
        master = master.merge(df_acs, on='CD_ID', how='left')

    # 5. 简单计算
    master = master.fillna(0)
    if 'Population' in master.columns and 'Housing_Units' in master.columns:
        # 避免除以0
        master['Trash_Per_Capita'] = master.apply(
            lambda x: x['Monthly_Trash_Tons'] / x['Population'] if x['Population'] > 0 else 0, axis=1)
        master['Rat_Density_Per_Unit'] = master.apply(
            lambda x: x['Rat_Complaints'] / x['Housing_Units'] if x['Housing_Units'] > 0 else 0, axis=1)

    # 保存
    filename = f"Manhattan_Data_{period_name}.csv"
    master.to_csv(filename, index=False)
    print(f"   ✅ 已生成: {filename}")


# ==========================================
# 4. 主程序
# ==========================================

# 加载公共资源
df_geo_base = load_geo()
df_trash_raw = pd.read_csv(FILE_TRASH_ALL)
df_acs_base = load_acs_features()

# --- 生成 Baseline (2017-2019) ---
# 注意：这里我们暂不放 ACS 数据，或者你可以决定是否要把 2023 的人口放进去作为参考
# 建议：Baseline 仅用于对比 老鼠 vs 垃圾 的关系，不做公平性分析，所以可以不放 ACS
build_period_dataset(
    "Baseline_2017_2019",
    FILE_RATS_OLD,
    "2017-01-01",
    "2019-12-31",
    df_geo_base,
    df_trash_raw,
    df_acs=None  # 不强行匹配旧人口
)

# --- 生成 Current (2023-2025) ---
# 这是你的主力数据集，必须包含所有特征
build_period_dataset(
    "Current_2023_2025",
    FILE_RATS_NEW,
    "2023-01-01",
    "2025-12-31",
    df_geo_base,
    df_trash_raw,
    df_acs=df_acs_base  # 放入 ACS 2023
)

print("\n🎉 全部处理完成！请检查生成的两个 CSV 文件。")