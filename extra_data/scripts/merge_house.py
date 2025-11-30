import pandas as pd
import re
import os

# ==========================================
# 1. 修正后的文件路径 (Strict Path Config)
# ==========================================
# 请确保脚本是在项目根目录下运行的
FILE_BASELINE = "extra_data/merged_data/Manhattan_Data_Baseline_2017_2019.csv"
FILE_CURRENT = "extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv"
FILE_HOUS = "extra_data/population_economy_data/Hous_1923_CDTA.xlsx"


# ==========================================
# 2. 核心工具
# ==========================================
def parse_id(val):
    s = str(val).upper().strip()
    nums = re.findall(r'\d+', s)
    if not nums: return None
    num = int(nums[-1])
    if 101 <= num <= 112: return num
    if 1 <= num <= 12: return 100 + num
    return None


def get_housing_data():
    print(f"🏠 正在从 {FILE_HOUS} 提取住房数据...")

    # 检查路径是否存在
    if not os.path.exists(FILE_HOUS):
        print(f"❌ 错误: 找不到文件！\n   请检查路径: {os.path.abspath(FILE_HOUS)}")
        return None

    try:
        # 显式指定 engine='openpyxl'
        df = pd.read_excel(FILE_HOUS, engine='openpyxl')

        # 1. 找 GeoID 列
        geo_cols = [c for c in df.columns if 'geo' in str(c).lower() and 'id' in str(c).lower()]
        if not geo_cols:
            print("❌ 未找到 GeoID 列")
            return None
        geo_col = geo_cols[0]

        # 2. 筛选曼哈顿
        df = df[df[geo_col].astype(str).str.startswith('MN')].copy()
        df['CD_ID'] = df[geo_col].apply(parse_id)

        # 3. 找住房单元列 (Total Housing Units)
        # 根据数据字典，Code是 HU1，Estimate 是 E -> 所以列名是 HU1E
        target_col = None

        # 优先找标准代码 'HU1E' (这是根据你字典确认的)
        if 'HU1E' in df.columns:
            target_col = 'HU1E'
        # 备选：有时候可能是 HU1
        elif 'HU1' in df.columns:
            target_col = 'HU1'
        # 再次备选：模糊搜索
        else:
            for c in df.columns:
                # 排除 'Occ' (Occupied), 找 'Total', 'Housing', 'Units'
                c_lower = str(c).lower()
                if 'hu' in c_lower and '1' in c_lower and 'e' in c_lower and 'occ' not in c_lower:
                    target_col = c
                    break

        if target_col:
            print(f"   ✅ 锁定住房列: [{target_col}]")
            return df[['CD_ID', target_col]].rename(columns={target_col: 'Housing_Units'})
        else:
            print("   ❌ 未找到住房单元列 (HU1E)，请检查 Excel 内容。")
            # 调试：打印前10个列名看看
            print(f"   前10个列名: {list(df.columns)[:10]}")
            return None

    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return None


def update_dataset(csv_file, df_hous):
    print(f"\n🔄 正在更新: {csv_file} ...")
    if not os.path.exists(csv_file):
        print(f"   ⚠️ 跳过 (文件不存在: {csv_file})")
        return

    df = pd.read_csv(csv_file)

    # 如果已经有 Housing_Units，先删掉避免重复列报错
    cols_to_drop = [c for c in ['Housing_Units', 'Housing_Density', 'Rats_Per_1k_Units'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 合并
    df = df.merge(df_hous, on='CD_ID', how='left')
    df['Housing_Units'] = df['Housing_Units'].fillna(0)

    # 计算密度指标
    # 1. 住房密度 (Units / sq km)
    # 面积转换: 1 sq ft = 9.2903e-8 sq km
    if 'SHAPE_Area' in df.columns:
        # 【修复步骤】检查并清洗数据：如果有逗号，先去掉逗号再转 float
        if df['SHAPE_Area'].dtype == 'object':
            print("   🔧 检测到 SHAPE_Area 包含逗号，正在清洗...")
            df['SHAPE_Area'] = df['SHAPE_Area'].astype(str).str.replace(',', '').astype(float)

        # 现在它是纯数字了，可以乘小数了
        df['Area_sqkm'] = df['SHAPE_Area'] * 9.2903e-8
        df['Housing_Density'] = df['Housing_Units'] / df['Area_sqkm']

    # 2. 住房老鼠密度 (Rats / 1000 Units)
    if 'Rat_Complaints' in df.columns:
        df['Rats_Per_1k_Units'] = df.apply(
            lambda x: (x['Rat_Complaints'] / x['Housing_Units'] * 1000) if x['Housing_Units'] > 0 else 0,
            axis=1
        )

    # 覆盖保存
    df.to_csv(csv_file, index=False)
    print(f"   ✅ 更新完成！新增列: Housing_Units, Housing_Density, Rats_Per_1k_Units")


# ==========================================
# 3. 执行
# ==========================================
if __name__ == "__main__":
    df_housing_clean = get_housing_data()

    if df_housing_clean is not None:
        update_dataset(FILE_BASELINE, df_housing_clean)
        update_dataset(FILE_CURRENT, df_housing_clean)
        print("\n🎉 所有文件已修补完成！")