import pandas as pd
import numpy as np

# ================= 1. 配置参数与约束 (L5 模型) =================
# 注意：请根据你的实际路径调整 INPUT_FILE
INPUT_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'
WORK_DAYS_PER_MONTH = 30.0

# 现实约束：考虑地理和运营损失 20%
NOMINAL_CAPACITY = 24.0
EFFICIENCY_LOSS_FACTOR = 0.20
DAILY_CAPACITY = NOMINAL_CAPACITY * (1 - EFFICIENCY_LOSS_FACTOR)  # 19.2 吨/天/车

# 拓扑分区定义 (用于 L5 模型，模拟相邻区共享)
POOLS = {
    'Lower': [101, 102, 103],
    'Midtown': [104, 105, 106, 107],
    'Uptown': [108, 109, 110, 111, 112]
}


# ================= 2. 核心函数 (已修改以适应 L5 约束) =================

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Rat_Complaints', 'Monthly_Trash_Tons', 'CD_ID'])

    # 频率决策 (策略核心)
    rat_threshold = df['Rat_Complaints'].median()
    df['Freq'] = df['Rat_Complaints'].apply(lambda x: 3 if x > rat_threshold else 2)

    # Tons per Pickup Day
    df['Tons_Per_Pickup'] = df['Monthly_Trash_Tons'] / 4.33 / df['Freq']

    # 添加区域划分 (L5 拓扑约束)
    def get_pool(cd_id):
        cd = int(cd_id)
        for pool_name, districts in POOLS.items():
            if cd in districts:
                return pool_name
        return 'Other'

    df['Pool'] = df['CD_ID'].apply(get_pool)
    return df


def optimize_schedule_sub(df_subset):
    """
    对一个数据子集 (可以是全局或局部池) 进行排班优化，找出最大负荷。
    这个函数是从你的 optimize_schedule 修改而来，现在用于局部和全局计算。
    """
    daily_loads = np.zeros(6)
    sorted_districts = df_subset.sort_values(by='Tons_Per_Pickup', ascending=False)

    for _, row in sorted_districts.iterrows():
        load = row['Tons_Per_Pickup']
        freq = row['Freq']

        best_pattern = None
        min_peak_load = float('inf')

        if freq == 3:
            options = [[0, 2, 4], [1, 3, 5]]
        else:
            options = [[0, 3], [1, 4], [2, 5]]

        for pattern in options:
            current_peak = max(daily_loads[day] + load for day in pattern)
            if current_peak < min_peak_load:
                min_peak_load = current_peak
                best_pattern = pattern

        if best_pattern is not None:
            for day in best_pattern:
                daily_loads[day] += load

    return max(daily_loads)


# ================= 3. 主程序运行与对比输出 =================
if __name__ == "__main__":

    df = load_and_prep_data(INPUT_FILE)

    # --- A. L4 模型：理想效率上限 (全局共享) ---
    global_max_load = optimize_schedule_sub(df)
    fleet_global_l4 = np.ceil(global_max_load / DAILY_CAPACITY)

    # --- B. L5 模型：现实拓扑约束下的解 ---
    total_fleet_l5 = 0
    pool_data = {}

    for pool_name, group in df.groupby('Pool'):
        if pool_name == 'Other': continue

        max_load_pool = optimize_schedule_sub(group)
        fleet_needed = np.ceil(max_load_pool / DAILY_CAPACITY)

        total_fleet_l5 += fleet_needed
        pool_data[pool_name] = {'Load': max_load_pool, 'Trucks': int(fleet_needed)}

    # --- 最终输出结果 ---
    print("\n=======================================================")
    print("=== L4 vs L5: 最终车队规模对比 (C_eff = 19.2 吨/天) ===")
    print("=======================================================")
    print(f"**理想基准 (L4 全局共享)**: {int(fleet_global_l4)} 辆")
    print(f"   (Max Load: {global_max_load:.1f} 吨)")

    print("\n--- L5 现实推荐 (拓扑约束) ---")
    for pool, data in pool_data.items():
        print(f"  {pool} 池 (局部瓶颈): {data['Load']:.1f} 吨 -> 需要 {data['Trucks']} 辆")

    print("-" * 45)
    print(f"**现实推荐总数 (L5 拓扑解): {int(total_fleet_l5)} 辆**")

    # 验证对比
    cost_increase = total_fleet_l5 - fleet_global_l4
    print(f"\n结论: 拓扑约束导致的增量成本: {int(cost_increase)} 辆车")
    print("=======================================================")

    # ================= 4. 导出 Q2 所需数据 =================
    # 构建要传给 Q2 的数据表
    output_df = df[
        ['CD_ID', 'DISTRICT', 'Freq', 'Tons_Per_Pickup', 'Rat_Complaints', 'Median_Income', 'Population']].copy()
    # 这里的 'Freq' 必须是你优化后的频率（高频区是3，低频区是2）
    output_df.to_csv('try/data/problem1_final_solution.csv', index=False)
    print("✅ 已导出 Q2 所需数据: problem1_final_solution.csv")