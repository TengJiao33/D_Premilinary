import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置 =================
SOLUTION_FILE = 'try/data/problem1_final_solution.csv'

# 参数假设
BIN_EFFECTIVENESS = 0.90  # 垃圾桶防鼠效率 (物理隔绝)
EFFICIENCY_LOSS_BAGS = 0.20  # 袋装效率损失 (慢)
EFFICIENCY_LOSS_BINS = 0.10  # 桶装效率损失 (快)
NOMINAL_CAPACITY = 24.0

# ================= 🌟 真实数据字典 (Based on MN.csv / PLUTO) =================
# 来源：User-generated from NYC PLUTO Dataset (2021)
# 含义：各 CD 中 "1-9 Unit Residential Buildings" 的占比
REAL_BIN_ADOPTION_STOPS = {
    101: 0.355,
    102: 0.493,  # 效率提升主战场
    103: 0.305,
    104: 0.305,
    105: 0.161,  # 最低，主要靠商用垃圾桶(不在此次住宅政策范围内)
    106: 0.384,
    107: 0.360,
    108: 0.395,
    109: 0.385,
    110: 0.483,  # 哈莱姆区，公平与效率的双赢
    111: 0.402,
    112: 0.153,  # 最低，必须依赖 Q4 策略
}


def load_data_robust():
    if not os.path.exists(SOLUTION_FILE):
        print(f"❌ 文件 {SOLUTION_FILE} 不存在")
        return None

    df = pd.read_csv(SOLUTION_FILE)
    df.columns = [c.strip() for c in df.columns]

    # 自动修复 CD_ID
    if 'CD_ID' not in df.columns:
        dist_col = next((c for c in df.columns if c.lower() == 'district'), None)
        if dist_col:
            def extract_id(d_str):
                try:
                    return int('1' + str(d_str).replace('MN', '').zfill(2))
                except:
                    return 0

            df['CD_ID'] = df[dist_col].apply(extract_id)

    # 自动修复 Monthly_Trash_Tons
    if 'Monthly_Trash_Tons' not in df.columns:
        if 'Tons_Per_Pickup' in df.columns and 'Freq' in df.columns:
            df['Monthly_Trash_Tons'] = df['Tons_Per_Pickup'] * df['Freq'] * 4.33

    return df


def run_analysis():
    df = load_data_robust()
    if df is None: return

    print("\n=== Step 1: 引入 2021 PLUTO 真实建筑数据 ===")
    # 1. 停靠点普及率 (影响卡车效率)
    df['Bin_Adoption_Stops'] = df['CD_ID'].map(REAL_BIN_ADOPTION_STOPS)

    # 2. 垃圾量普及率 (影响老鼠)
    # 修正因子：小楼垃圾量少，设为停靠点比例的 40% (Volume Weighted)
    df['Bin_Adoption_Volume'] = df['Bin_Adoption_Stops'] * 0.4

    avg_stops = df['Bin_Adoption_Stops'].mean()
    print(f"平均停靠点普及率 (Trucks): {avg_stops:.1%} (基于真实数据)")
    print(f"平均垃圾量覆盖率 (Rats): {df['Bin_Adoption_Volume'].mean():.1%}")

    print("\n=== Step 2: 对老鼠的影响 (叠加效应) ===")
    df['Rats_After_Bins'] = df['Rat_Complaints'] * (1 - df['Bin_Adoption_Volume'] * BIN_EFFECTIVENESS)
    reduction_pct = (df['Rat_Complaints'].sum() - df['Rats_After_Bins'].sum()) / df['Rat_Complaints'].sum()
    print(f"✅ 仅因垃圾桶政策，老鼠预计减少: {reduction_pct:.1%}")

    print("\n=== Step 3: 对车队的影响 (效率飞跃) ===")
    # 效率公式
    df['New_Efficiency_Loss'] = (EFFICIENCY_LOSS_BAGS * (1 - df['Bin_Adoption_Stops'])) + \
                                (EFFICIENCY_LOSS_BINS * df['Bin_Adoption_Stops'])
    df['New_Daily_Capacity'] = NOMINAL_CAPACITY * (1 - df['New_Efficiency_Loss'])

    # 重新计算需求 (L5 基准: 142辆)
    total_capacity_needed = 142 * 19.2
    avg_new_cap = df['New_Daily_Capacity'].mean()

    print(f"平均单车有效日运力从 19.20 吨 -> {avg_new_cap:.2f} 吨")

    new_fleet_size = int(np.ceil(total_capacity_needed / avg_new_cap))
    saved_trucks = 142 - new_fleet_size

    print(f"🚚 车队规模变化: 142 辆 -> {new_fleet_size} 辆")
    print(f"💰 节省车辆: {saved_trucks} 辆")

    # 绘图
    plt.figure(figsize=(7, 6))
    bars = plt.bar(['Current L5', 'With Bins Q5'], [142, new_fleet_size],
                   color=['#1f77b4', '#2ca02c'], alpha=0.8, width=0.5)

    plt.text(0, 142 + 2, "142", ha='center', fontweight='bold', fontsize=12)
    plt.text(1, new_fleet_size + 2, str(new_fleet_size), ha='center', fontweight='bold', fontsize=12)

    plt.title('Impact of Bin Adoption on Fleet Size (Real 2021 Data)', fontsize=14)
    plt.ylabel('Trucks Needed')
    plt.grid(axis='y', alpha=0.3)

    plt.savefig('try/image/Viz_Q5_RealData_Impact.png', dpi=300)
    print("📊 结果图已保存: Viz_Q5_RealData_Impact.png")

    df.to_csv('try/data/problem5_real_data_result.csv', index=False)


if __name__ == "__main__":
    run_analysis()