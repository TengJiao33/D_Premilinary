import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置 =================
SOLUTION_FILE = 'try/data/problem1_final_solution.csv'
FLEET_SIZE = 142  # 我们的 L5 模型结果
NOMINAL_CAPACITY = 24.0  # 理论最大运力 (2 trips)
BUFFER_CAPACITY = 19.2  # 我们平时排班用的运力 (20% buffer)


# ================= 1. 加载数据 =================
def load_data():
    if not os.path.exists(SOLUTION_FILE):
        print("请先运行 Q1 代码生成数据！")
        return None
    return pd.read_csv(SOLUTION_FILE)


# ================= 2. 压力测试引擎 =================
def stress_test(df, failure_rate=0.0, load_spike=0.0, weather_impact=0.0):
    """
    模拟一天的运营状况
    :param failure_rate: 车辆故障率 (0.0 - 0.5)
    :param load_spike: 垃圾激增比例 (0.0 - 0.5)
    :param weather_impact: 天气导致的额外效率损失 (0.0 - 0.5)
    :return: 剩余未收集垃圾量 (Tons), 崩溃的社区数量
    """
    # 1. 供给侧冲击 (Supply Shock)
    # 实际可用车辆
    available_trucks = int(FLEET_SIZE * (1 - failure_rate))
    # 实际单车运力 (受天气影响)
    # 基础模型已经扣了0.2，天气会再扣
    current_efficiency = (1 - 0.2) * (1 - weather_impact)
    actual_capacity_per_truck = NOMINAL_CAPACITY * current_efficiency

    total_capacity = available_trucks * actual_capacity_per_truck

    # 2. 需求侧冲击 (Demand Shock)
    # 假设今天是负荷最大的一天 (Worst Case from Q1)
    # Q1算出最大负荷约 2161.5 吨。我们用这个基准加 spikes
    base_load = 2161.5
    total_load = base_load * (1 + load_spike)

    # 3. 结果计算
    uncollected = max(0, total_load - total_capacity)
    success_rate = min(1.0, total_capacity / total_load)

    return success_rate, uncollected


# ================= 3. 适应性策略模拟 (加班模式) =================
def adaptive_strategy_test(df, load_spike):
    """
    模拟：如果不加车，而是开启 '加班模式' (Overtime, R=2.5 trips/day)
    能否扛住垃圾激增？
    """
    base_load = 2161.5
    total_load = base_load * (1 + load_spike)

    # 标准模式 (2 trips, 20% loss) -> 19.2 tons/truck
    cap_std = FLEET_SIZE * 19.2

    # 加班模式 (加开0.5趟, 效率略降) -> 假设 2.5 trips * 12 * 0.8 = 24 tons/truck
    cap_ot = FLEET_SIZE * 24.0

    return (cap_std >= total_load), (cap_ot >= total_load)


# ================= 4. 绘图与分析 =================
def run_analysis(df):
    print("=== Q3: 鲁棒性与中断场景分析 ===")

    # --- 场景 A: 混合压力测试矩阵 ---
    # X轴: 车辆故障率, Y轴: 垃圾激增率
    # 值: 服务成功率 (0-100%)

    failures = np.linspace(0, 0.3, 10)  # 0% 到 30% 故障
    spikes = np.linspace(0, 0.3, 10)  # 0% 到 30% 激增

    heatmap_data = np.zeros((10, 10))

    for i, f in enumerate(failures):
        for j, s in enumerate(spikes):
            rate, _ = stress_test(df, failure_rate=f, load_spike=s)
            heatmap_data[j, i] = rate  # 注意行列对应

    # 绘图 1: 热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".0%", cmap="RdYlGn",
                xticklabels=[f"{x:.0%}" for x in failures],
                yticklabels=[f"{y:.0%}" for y in spikes],
                vmin=0.8, vmax=1.0)

    plt.xlabel('Vehicle Breakdown Rate')
    plt.ylabel('Waste Spike Rate')
    plt.title('Robustness Heatmap: Service Level under Stress')
    plt.tight_layout()
    plt.savefig('try/image/Viz_Q3_Robustness_Heatmap.png', dpi=300)
    print("📊 鲁棒性热力图已保存: Viz_Q3_Robustness_Heatmap.png")

    # --- 场景 B: 极端天气适应性 (加班策略) ---
    spike_range = np.linspace(0, 0.5, 50)  # 0% 到 50% 激增
    std_res = []
    ot_res = []

    for s in spike_range:
        std_ok, ot_ok = adaptive_strategy_test(df, s)
        std_res.append(1 if std_ok else 0)  # 1=Survive, 0=Fail
        ot_res.append(1 if ot_ok else 0)

    # 找到崩溃临界点
    limit_std = next((s for s, ok in zip(spike_range, std_res) if ok == 0), 0.5)
    limit_ot = next((s for s, ok in zip(spike_range, ot_res) if ok == 0), 0.5)

    print(f"\n[压力测试结论]")
    print(f"1. 标准模式 (Standard) 崩溃阈值: 垃圾激增 > {limit_std:.1%}")
    print(f"2. 加班模式 (Overtime) 崩溃阈值: 垃圾激增 > {limit_ot:.1%}")
    print(f"-> 策略建议: 遇到 >{limit_std:.1%} 的激增时，立即启动加班预案。")

    # 绘图 2: 适应性生存曲线
    plt.figure(figsize=(10, 5))
    # 简单的区域填充图
    plt.fill_between(spike_range * 100, 0, std_res, color='red', alpha=0.3, label='Standard Capacity')
    plt.fill_between(spike_range * 100, 0, ot_res, color='green', alpha=0.3, label='With Adaptive Overtime')

    plt.axvline(limit_std * 100, color='red', linestyle='--', label=f'Std Limit ({limit_std:.0%})')
    plt.axvline(limit_ot * 100, color='green', linestyle='--', label=f'OT Limit ({limit_ot:.0%})')

    plt.xlabel('Unexpected Waste Spike (%)')
    plt.ylabel('System Survival (1=OK, 0=Collapse)')
    plt.title('Adaptation Strategy: Extending Limits with Overtime')
    plt.legend()
    plt.tight_layout()
    plt.savefig('try/image/Viz_Q3_Adaptation.png', dpi=300)
    print("📊 适应性分析图已保存: Viz_Q3_Adaptation.png")


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        run_analysis(df)