import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# ================= 配置 =================
DATA_PATH = os.path.join('..', 'extra_data', 'merged_data', 'Manhattan_Data_Current_2023_2025.csv')
# 这里读取的是你 v11 版本生成的那个最优时间表
OPTIMIZED_RESULT_PATH = 'problem1_final_result_v11.csv'
OUTPUT_CHART = 'problem1_savings_v2_chart.png'
OUTPUT_REPORT = 'problem1_savings_v2_report.txt'

TRUCK_CAPACITY = 12.0 * 0.9  # 10.8 吨
STREET_CAPACITY_TONS = 600.0 # 物理容量死线 (与 v11 保持一致)

def load_data(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"Missing: {filepath}")
    df = pd.read_csv(filepath)
    districts = []
    for _, row in df.iterrows():
        cd_id = int(row['CD_ID'])
        name = f"MN{cd_id % 100:02d}"
        daily = row['Monthly_Trash_Tons'] / 30.0
        districts.append({'id': name, 'daily': daily})
    return districts

def calculate_dedicated_fleet(districts):
    """
    计算【局部优化/不可跨区】情形下的车队总数。
    逻辑：每个区自己算需要几辆车，然后简单相加。
    因为车不能跨区借用，所以必须满足每个区自己的最大需求。
    """
    total_dedicated_fleet = 0
    details = []
    
    print(f"{'District':<10} | {'Daily':<8} | {'Freq':<5} | {'Trucks (Dedicated)':<20}")
    print("-" * 60)
    
    for d in districts:
        daily = d['daily']
        
        # 1. 局部优化：自己算 2次 还是 3次
        # 如果间隔 3.5 天(2次) 会爆仓，就选 3次；否则为了省钱选 2次
        if (daily * 3.5) > STREET_CAPACITY_TONS:
            freq = 3
        else:
            freq = 2
            
        # 2. 计算该区需要的单次车数
        # 这就是该区必须配备的"固定资产"
        tons_per_visit = daily * 7 / freq
        trucks_needed = math.ceil(tons_per_visit / TRUCK_CAPACITY)
        
        total_dedicated_fleet += trucks_needed
        
        details.append({
            'id': d['id'],
            'trucks': trucks_needed
        })
        print(f"{d['id']:<10} | {daily:<8.1f} | {freq:<5} | {trucks_needed:<20}")
        
    return total_dedicated_fleet, details

def get_shared_fleet_peak(optimized_csv_path):
    """
    获取【全局优化/可跨区】情形下的车队峰值。
    直接读取 v11 算出来的 Total_Trucks 行的最大值。
    """
    if not os.path.exists(optimized_csv_path):
        print("[Error] 找不到优化结果，请先运行 v11 脚本！")
        return 0, []
        
    df = pd.read_csv(optimized_csv_path, index_col=0)
    
    # 提取最后一行 Total_Trucks (除去最后一列 Frequency)
    if 'Total_Trucks' not in df.index:
        print("[Error] CSV 中缺少 Total_Trucks 行")
        return 0, []
        
    # 获取 Mon-Sun 的数据
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_loads = df.loc['Total_Trucks', days].values.astype(int)
    
    # 共享模式下的车队规模 = 这一周里最忙那一天的车数
    # (因为车是共享的，只要满足峰值即可)
    peak_fleet = np.max(daily_loads)
    
    return peak_fleet, daily_loads

def main():
    # 1. 加载基础数据
    districts_data = load_data(DATA_PATH)
    
    # 2. 场景 A：计算独立车队总和 (不可跨区)
    dedicated_fleet_size, dedicated_details = calculate_dedicated_fleet(districts_data)
    
    # 3. 场景 B：获取共享车队峰值 (可跨区，v11结果)
    shared_fleet_size, daily_curve = get_shared_fleet_peak(OPTIMIZED_RESULT_PATH)
    
    if shared_fleet_size == 0: return # 出错中止
    
    # 4. 计算节省
    savings = dedicated_fleet_size - shared_fleet_size
    percent = (savings / dedicated_fleet_size) * 100
    
    # 5. 生成报告文本
    report = f"""
==================================================================
Problem D - Q1 Final Analysis: Dedicated vs. Shared Fleet
==================================================================

[Scenario A: Local Optimization / Siloed Dispatch]
- Description: Each district determines its optimal frequency (2x or 3x) 
  based on its own trash generation and capacity limits.
- Constraint: Trucks CANNOT be shared between districts (No Cross-Scheduling).
- Calculation: Sum of trucks required by each district individually.
- TOTAL FLEET REQUIRED: {dedicated_fleet_size} Trucks

[Scenario B: Global Optimization / Collaborative Dispatch]
- Description: Districts share a centralized fleet. Schedules are staggered 
  (Mon-Sun) to balance the daily load across the entire island.
- Constraint: Trucks CAN be rescheduled to different districts on different days.
- Calculation: The maximum number of trucks active on the busiest day.
- TOTAL FLEET REQUIRED: {shared_fleet_size} Trucks

[CONCLUSION]
- Savings in Capital Expenditure (Fleet Size): {savings} Trucks
- Efficiency Gain: {percent:.1f}%
- Key Driver: "Temporal Load Balancing" - utilizing the idle capacity of trucks
  on non-peak days by rotating them between districts.
"""
    print(report)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
        
    # 6. 画图 (直观展示差距)
    # 我们画一条横线表示"独立车队总数"，画柱状图表示"共享车队每日需求"
    plt.figure(figsize=(10, 6))
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    bars = plt.bar(days, daily_curve, color='#4c72b0', alpha=0.8, label='Optimized Daily Demand (Shared)')
    
    # 画红线：独立车队总数
    plt.axhline(y=dedicated_fleet_size, color='#c44e52', linewidth=3, linestyle='--', label='Baseline Fleet Size (Dedicated)')
    
    # 画绿线：优化后的车队规模 (即蓝色柱子的最高点)
    plt.axhline(y=shared_fleet_size, color='#55a868', linewidth=2, linestyle='-', label=f'Optimized Fleet Size ({shared_fleet_size})')
    
    # 标注
    plt.text(0, dedicated_fleet_size + 2, f'Scenario A: {dedicated_fleet_size} Trucks', color='#c44e52', fontweight='bold')
    plt.text(6, shared_fleet_size + 2, f'Scenario B: {shared_fleet_size} Trucks', color='#55a868', fontweight='bold', ha='right')
    
    plt.title('Fleet Savings: Dedicated (Local) vs. Shared (Global) Dispatch')
    plt.ylabel('Number of Trucks Required')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
    print(f"[Chart] 图表已生成: {OUTPUT_CHART}")

if __name__ == "__main__":
    main()