import pandas as pd
import numpy as np

# ================= 配置参数 =================
INPUT_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'
TRUCK_CAPACITY_PER_TRIP = 12.0  # 单次运力 (吨)
TRIPS_PER_DAY = 1  # 每日趟数
DAILY_CAPACITY = TRUCK_CAPACITY_PER_TRIP * TRIPS_PER_DAY  # 24吨


# ================= 1. 数据加载与频率决策 =================
def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)

    # --- 策略：决定回收频率 ---
    # 逻辑：鼠患投诉量高于中位数的，定为高频区 (3次/周)，否则为标准区 (2次/周)
    rat_threshold = df['Rat_Complaints'].median()

    def get_freq_and_label(row):
        if row['Rat_Complaints'] > rat_threshold:
            return 3, 'High(3x)'
        else:
            return 2, 'Std(2x)'

    df[['Freq', 'Label']] = df.apply(lambda x: pd.Series(get_freq_and_label(x)), axis=1)

    # 计算“单次回收日的垃圾量” (Tons per Pickup Day)
    # 假设一个月 4.33 周
    df['Tons_Per_Pickup'] = df['Monthly_Trash_Tons'] / 4.33 / df['Freq']

    return df


# ================= 2. 智能排班 (核心优化) =================
def optimize_schedule(df):
    """
    将社区分配到具体的“工作日模式”，以平衡每天的垃圾总量。
    模式池：
    - Freq=3: [Mon, Wed, Fri] 或 [Tue, Thu, Sat]
    - Freq=2: [Mon, Thu] 或 [Tue, Fri] 或 [Wed, Sat]
    """
    # 初始化一周6天的垃圾负荷 (周日休息)
    # 0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat
    daily_loads = np.zeros(6)

    # 记录每个区的排班结果
    schedule_map = {}  # CD_ID -> [Days]

    # 贪心算法：优先安排垃圾量最大的区
    sorted_districts = df.sort_values(by='Tons_Per_Pickup', ascending=False)

    for _, row in sorted_districts.iterrows():
        load = row['Tons_Per_Pickup']
        freq = row['Freq']
        cd_id = row['CD_ID']

        best_pattern = None
        min_peak_load = float('inf')

        # 定义可选的时间模式
        if freq == 3:
            options = [[0, 2, 4], [1, 3, 5]]  # MWF vs TTS
        else:
            options = [[0, 3], [1, 4], [2, 5]]  # MTh vs TF vs WS

        # 尝试每一种模式，看谁能让“全周最大峰值”最小
        for pattern in options:
            current_peak = max(daily_loads[day] + load for day in pattern)
            # 同时也考虑当天的总负荷，稍微倾向于填补空闲日
            if current_peak < min_peak_load:
                min_peak_load = current_peak
                best_pattern = pattern

        # 执行分配
        for day in best_pattern:
            daily_loads[day] += load
        schedule_map[cd_id] = best_pattern

    print("\n--- 智能排班后的每日垃圾总量 (Tons) ---")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    for i, load in enumerate(daily_loads):
        print(f"{days[i]}: {load:.1f} 吨")

    max_daily_load = max(daily_loads)
    print(f"\n全周最大单日负荷: {max_daily_load:.1f} 吨 (决定了车队规模)")

    return max_daily_load, schedule_map


# ================= 3. 计算车队 (Bin Packing) =================
# 基于最大单日负荷计算
def calculate_fleet_size(max_daily_load):
    # 理论最小车数 (无视地理分割，理想拼单)
    # 因为我们已经在 schedule 层面做了平衡，每天的总量是我们要处理的对象
    # 这里应用 Bin Packing 思想：总垃圾 / 单车运力

    # 考虑到实际可能有碎块，稍微加一点余量，或者模拟具体的一天
    # 简单估算：直接除以运力，向上取整。因为跨区拼单允许我们把剩余的填满。
    trucks_needed = np.ceil(max_daily_load / DAILY_CAPACITY)

    return int(trucks_needed)


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载数据
    df = load_and_prep_data(INPUT_FILE)
    print(f"数据加载完毕，共有 {len(df[df['Freq'] == 3])} 个高频区 (3x), {len(df[df['Freq'] == 2])} 个标准区 (2x)")

    # 2. 之前的笨办法 (所有人在周一收)
    worst_case_load = df['Tons_Per_Pickup'].sum()
    worst_case_trucks = np.ceil(worst_case_load / DAILY_CAPACITY)
    print(f"\n[Baseline] 同步回收 (所有区同一天): 需要 {int(worst_case_trucks)} 辆车")

    # 3. 现在的聪明办法 (错峰排班)
    max_load_balanced, schedules = optimize_schedule(df)

    # 4. 计算最终车队
    optimized_trucks = calculate_fleet_size(max_load_balanced)

    print(f"\n[Optimized] 错峰排班 + 跨区共享: 需要 {optimized_trucks} 辆车")

    # 5. 结论
    saved = worst_case_trucks - optimized_trucks
    print(f"\nResult: 通过排班优化，节省了 {int(saved)} 辆车 ({saved / worst_case_trucks:.1%})")

    # 导出排班表供论文使用
    print("\n--- 排班表示例 ---")
    days_lookup = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    for cd_id, pattern in list(schedules.items())[:5]:
        d_str = ",".join([days_lookup[d] for d in pattern])
        print(f"CD {int(cd_id)}: {d_str}")