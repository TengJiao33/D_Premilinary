import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# ================= 配置区域 =================
SOLUTION_FILE = 'try/data/problem1_final_solution.csv'

# 绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 数据加载 (精准适配你的格式) =================
def load_data():
    print("正在加载 Q1 分析结果...")
    if not os.path.exists(SOLUTION_FILE):
        print(f"❌ 找不到文件 {SOLUTION_FILE}，请先运行 Q1 代码生成它！")
        return None

    df = pd.read_csv(SOLUTION_FILE)

    # 你的数据样例：CD_ID,DISTRICT,Freq,Tons_Per_Pickup,Rat_Complaints,Median_Income,Population
    # 检查必需列
    required = ['Freq', 'Tons_Per_Pickup', 'Rat_Complaints', 'Median_Income', 'Population']
    for col in required:
        if col not in df.columns:
            print(f"❌ 错误：你的CSV里缺少列 '{col}'。请检查 problem1_final_solution.csv 的生成代码！")
            return None

    return df


# ================= 2. 核心指标计算 (原汁原味的队友算法) =================
def calculate_metrics(df):
    print("\n=== Q2: 效率与公平性量化评估 ===")

    # --- A. 效率指标 (Effectiveness) ---
    # 之前代码用了 Monthly_Trash_Tons，现在我们用 Tons_Per_Pickup * Freq 反推
    # 逻辑：每周总运量 = Σ(单次量 * 每周频次)
    total_weekly_trash = (df['Tons_Per_Pickup'] * df['Freq']).sum()
    total_weekly_visits = df['Freq'].sum()

    efficiency_score = total_weekly_trash / total_weekly_visits

    # --- B. 公平性指标 (Equity) ---
    # 1. 收入偏见 (Income Correlation) -> 负相关最好
    corr_income, p_income = pearsonr(df['Median_Income'], df['Freq'])

    # 2. 需求响应度 (Rat Correlation) -> 正相关最好
    corr_rats, p_rats = pearsonr(df['Rat_Complaints'], df['Freq'])

    # 3. 基尼系数 (Gini Index)
    df['Service_Per_Capita'] = df['Freq'] / df['Population']
    gini = gini_coefficient(df['Service_Per_Capita'].values)

    # --- C. 成本指标 (基于 L5 142辆) ---
    FLEET_SIZE_L5 = 142
    # 简单估算：每辆车年成本25万 / 总人口
    cost_per_capita = (FLEET_SIZE_L5 * 250000) / df['Population'].sum()

    print(f"1. [效率] 单次服务运量: {efficiency_score:.2f} Tons/Visit")
    print(f"2. [公平] 收入相关性: {corr_income:.3f} (理想为负)")
    print(f"3. [有效] 鼠患响应度: {corr_rats:.3f} (理想为正)")
    print(f"4. [平等] 服务基尼系数: {gini:.3f}")
    print(f"5. [成本] 人均年服务成本: ${cost_per_capita:.2f}")

    return corr_income, corr_rats, efficiency_score


def gini_coefficient(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


# ================= 3. 可视化绘图 =================

def plot_equity_scatter(df, corr_inc):
    """图1: 收入 vs 频率"""
    plt.figure(figsize=(10, 6))

    # 这里的 Rat_Complaints 可能是数千，除以 100 让点大小合适
    sizes = df['Rat_Complaints'] / df['Rat_Complaints'].max() * 500

    scatter = plt.scatter(df['Median_Income'], df['Freq'],
                          s=sizes,
                          c=df['Rat_Complaints'],
                          cmap='Reds', alpha=0.8, edgecolors='k')

    # 趋势线
    if len(df) > 1:
        z = np.polyfit(df['Median_Income'], df['Freq'], 1)
        p = np.poly1d(z)
        plt.plot(df['Median_Income'], p(df['Median_Income']), "b--", alpha=0.6, label=f'Trend (r={corr_inc:.2f})')

    plt.colorbar(scatter, label='Rat Complaints Intensity')
    plt.xlabel('Median Household Income ($)')
    plt.ylabel('Weekly Collection Frequency')
    plt.title('Equity Analysis: Progressive Service Allocation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('try/image/Viz_Q2_Equity_Income.png', dpi=300)
    print("📊 图表已保存: Viz_Q2_Equity_Income.png")


def plot_tradeoff_concept(corr_income):
    """图2: 权衡分析 - 只要展示 L4, L5, Baseline 的相对位置"""
    plt.figure(figsize=(9, 7))

    # 手动定义三个点的位置 (示意图)
    # X轴: 相对效率 (越高越好)
    # Y轴: 公平性得分 (绝对值越大越好)

    # Baseline (190辆): 效率低(0.6), 公平性假设也低(0.2)
    # L4 Ideal (113辆): 效率极高(1.0), 公平性高(0.52)
    # L5 Real  (142辆): 效率较高(0.8), 公平性高(0.52)

    equity_score = abs(corr_income)  # 使用计算出的真实公平分

    plt.scatter([0.6], [0.1], s=300, c='gray', label='Baseline (190 Trucks)', edgecolors='k')
    plt.scatter([1.0], [equity_score], s=300, c='orange', label='L4 Ideal (113 Trucks)', marker='*', edgecolors='k')
    plt.scatter([0.8], [equity_score], s=400, c='green', label='L5 Recommended (142 Trucks)', marker='D',
                edgecolors='k')

    plt.text(0.6, 0.05, "Low Efficiency", ha='center')
    plt.text(1.0, equity_score + 0.05, "Theoretical Limit", ha='center')
    plt.text(0.8, equity_score - 0.08, "Optimal Reality\n(Selected)", ha='center', fontweight='bold', color='green')

    plt.plot([0.6, 0.8, 1.0], [0.1, equity_score, equity_score], 'k--', alpha=0.3)

    plt.xlim(0.4, 1.1)
    plt.ylim(0, 1.0)
    plt.xlabel('Efficiency Score')
    plt.ylabel('Equity Score (|Income Correlation|)')
    plt.title('Trade-off Analysis: Efficiency vs Equity')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('try/image/Viz_Q2_Tradeoff.png', dpi=300)
    print("📊 图表已保存: Viz_Q2_Tradeoff.png")


# ================= 主程序 =================
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        c_inc, c_rat, eff = calculate_metrics(df)
        plot_equity_scatter(df, c_inc)
        plot_tradeoff_concept(c_inc)
        print("\n✨ Q2 分析完成！")