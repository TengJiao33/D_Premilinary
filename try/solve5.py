import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置 =================
# 读取 Q1 结果 (作为基准)
SOLUTION_FILE = 'try/data/problem1_final_solution.csv'
# 读取原始数据 (为了获取 Area 和 Housing Units)
RAW_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'

# 参数假设
BIN_EFFECTIVENESS = 0.90  # 垃圾桶对老鼠的隔绝效率 (90%)
EFFICIENCY_LOSS_BAGS = 0.20  # 袋装垃圾的效率损失 (20%)
EFFICIENCY_LOSS_BINS = 0.10  # 桶装垃圾的效率损失 (10%, 更快)
NOMINAL_CAPACITY = 24.0  # 24吨理论运力

# L5 模型分区 (用于重新计算车队)
POOLS = {
    'Lower': [101, 102, 103],
    'Midtown': [104, 105, 106, 107],
    'Uptown': [108, 109, 110, 111, 112]
}


# ================= 1. 数据加载与融合 =================
def load_and_merge_data():
    if not os.path.exists(SOLUTION_FILE) or not os.path.exists(RAW_FILE):
        print("❌ 文件缺失！")
        return None

    df_sol = pd.read_csv(SOLUTION_FILE)
    df_raw = pd.read_csv(RAW_FILE)

    # 我们需要 Raw 里的 Housing_Units 和 SHAPE_Area
    # 合并
    df = pd.merge(df_sol, df_raw[['CD_ID', 'Housing_Units', 'SHAPE_Area']], on='CD_ID', how='left')

    # 再次检查 Monthly_Trash_Tons
    if 'Monthly_Trash_Tons' not in df.columns:
        df['Monthly_Trash_Tons'] = df['Tons_Per_Pickup'] * df['Freq'] * 4.33

    return df


# ================= 2. 估算垃圾桶普及率 (Bin Adoption) =================
def estimate_bin_adoption(df):
    print("\n=== Step 1: Estimating Bin Adoption Rate ===")

    # 逻辑：住房密度越高 -> 大楼越多 -> 1-9户小楼比例越低 -> 普及率越低
    # 计算密度 (Units per sq km)
    # SHAPE_Area is usually sq feet. 1 sq ft = 9.29e-8 sq km
    df['Area_sqkm'] = df['SHAPE_Area'] * 9.2903e-8
    df['Housing_Density'] = df['Housing_Units'] / df['Area_sqkm']

    # 建立模型：
    # 设全曼哈顿平均普及率为 25% (题目说 NYC 是 41%，曼哈顿显著低)
    # 使用反比函数缩放

    # 归一化密度 (0-1), 密度越低 score 越高
    max_dens = df['Housing_Density'].max()
    min_dens = df['Housing_Density'].min()

    # 线性插值：密度最低的区(Low Density) -> 假设 45% 普及
    # 密度最高的区(High Density) -> 假设 10% 普及

    def get_rate(density):
        # 归一化位置 (0 = 最低密, 1 = 最高密)
        pos = (density - min_dens) / (max_dens - min_dens)
        # 线性映射到 0.45 -> 0.10
        rate = 0.45 - (pos * (0.45 - 0.10))
        return rate

    df['Bin_Adoption_Rate'] = df['Housing_Density'].apply(get_rate)

    print(f"曼哈顿平均垃圾桶普及率: {df['Bin_Adoption_Rate'].mean():.1%}")
    print("各区普及率预估:")
    print(df[['DISTRICT', 'Bin_Adoption_Rate']].sort_values(by='Bin_Adoption_Rate', ascending=False).head(3))
    print("...")

    return df


# ================= 3. 计算对老鼠的影响 =================
def calculate_rat_impact(df):
    print("\n=== Step 2: Impact on Rats ===")

    # 公式：New Rats = Current * (1 - Adoption * Effectiveness)
    # 这叠加在 Q4 的优化之上吗？题目问的是 "new rule affect"，我们基于当前现状分析
    # 我们可以展示：如果加上 Q4 策略，效果会叠加

    df['Rats_After_Bins'] = df['Rat_Complaints'] * (1 - df['Bin_Adoption_Rate'] * BIN_EFFECTIVENESS)

    reduction_pct = (df['Rat_Complaints'].sum() - df['Rats_After_Bins'].sum()) / df['Rat_Complaints'].sum()
    print(f"仅因垃圾桶政策，老鼠预计减少: {reduction_pct:.1%}")

    return df


# ================= 4. 计算对卡车的影响 (L5+ 模型) =================
def calculate_truck_impact(df):
    print("\n=== Step 3: Impact on Fleet Efficiency ===")

    # 1. 计算新的加权效率损失因子
    # Loss = (Loss_Bag * (1-Rate)) + (Loss_Bin * Rate)
    df['New_Efficiency_Loss'] = (EFFICIENCY_LOSS_BAGS * (1 - df['Bin_Adoption_Rate'])) + \
                                (EFFICIENCY_LOSS_BINS * df['Bin_Adoption_Rate'])

    # 2. 计算新的单车日运力
    df['New_Daily_Capacity'] = NOMINAL_CAPACITY * (1 - df['New_Efficiency_Loss'])

    print(f"平均单车日运力从 19.2 吨提升至: {df['New_Daily_Capacity'].mean():.2f} 吨")

    # 3. 重新计算 L5 车队规模 (Pool based)

    # 先重新计算排班后的最大负荷 (需要引入之前的 optimize 逻辑，这里简化处理)
    # 假设排班负荷与 Q1 相同 (Tons_Per_Pickup 不变)，只改变 Capacity
    # 我们直接读取 Q1 算出的局部最大负荷。
    # 为了严谨，我们重新跑一遍简单的 Bin Packing

    total_new_fleet = 0
    df['Pool'] = df['CD_ID'].apply(lambda x: next((k for k, v in POOLS.items() if x in v), 'Other'))

    # 这里我们做一个简化：利用 Q1 的 Load 结果。
    # 如果 Q1 L5 算出总 Capacity 需求。
    # 下面我们模拟计算：

    fleet_comparison = []

    for pool, group in df.groupby('Pool'):
        if pool == 'Other': continue

        # 估算该池的最大日负荷 (近似值，假设排班平衡度不变)
        # 用 Q1 的逻辑: Monthly / 30 / Capacity? No.
        # 直接用 Q1 算出的 Trucks * 19.2 倒推 Load? 不太准。
        # 最好是重新算一下需求车辆 = ceil(Daily_Load / New_Average_Capacity_of_Pool)

        # 我们用一个简单的近似：Monthly_Trash / 30 得到日均，乘一个峰值因子 1.2 (经验值)
        # 或者更简单：New_Trucks = Old_Trucks * (Old_Cap / New_Cap)

        pool_avg_cap = group['New_Daily_Capacity'].mean()
        old_cap = 19.2
        ratio = old_cap / pool_avg_cap

        # 这是一个估算，展示趋势
        # 假设 Q1 L5 中 Lower=40, Midtown=51, Uptown=51 (Total 142)
        if pool == 'Lower':
            base_trucks = 40
        elif pool == 'Midtown':
            base_trucks = 51
        elif pool == 'Uptown':
            base_trucks = 51
        else:
            base_trucks = 0

        new_trucks = base_trucks * ratio
        # 向上取整
        new_trucks_int = int(np.ceil(new_trucks))

        fleet_comparison.append({
            'Pool': pool,
            'Old_Trucks': base_trucks,
            'New_Trucks': new_trucks_int,
            'Adoption_Rate': group['Bin_Adoption_Rate'].mean()
        })
        total_new_fleet += new_trucks_int

    res_df = pd.DataFrame(fleet_comparison)
    print("\n车队规模变化预测:")
    print(res_df)
    print(f"\n总车队规模: 142 辆 -> {total_new_fleet} 辆")
    print(f"节省车辆: {142 - total_new_fleet} 辆")

    return total_new_fleet, res_df


# ================= 5. 可视化 =================
def plot_results(df, old_fleet, new_fleet):
    # 图1: 车队对比
    plt.figure(figsize=(8, 6))
    plt.bar(['Current (L5)', 'With Bins (Q5)'], [old_fleet, new_fleet],
            color=['#1f77b4', '#2ca02c'], alpha=0.8, width=0.5)

    for i, v in enumerate([old_fleet, new_fleet]):
        plt.text(i, v + 2, str(v), ha='center', fontsize=12, fontweight='bold')

    plt.title('Impact of "Bins not Bags" on Fleet Size', fontsize=14)
    plt.ylabel('Number of Trucks Needed')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('Viz_Q5_Fleet_Impact.png', dpi=300)
    print("📊 车队影响图已保存: Viz_Q5_Fleet_Impact.png")


# ================= 主程序 =================
if __name__ == "__main__":
    df = load_and_merge_data()
    if df is not None:
        df = estimate_bin_adoption(df)
        df = calculate_rat_impact(df)
        new_fleet, res_df = calculate_truck_impact(df)
        plot_results(df, 142, new_fleet)

        # 导出结果
        df.to_csv('problem5_bins_impact.csv', index=False)
        print("\n✅ Q5 分析完成！")