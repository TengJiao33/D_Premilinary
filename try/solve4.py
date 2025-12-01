import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# ================= 配置 =================
INPUT_FILE = 'try/data/problem1_final_solution.csv'
# 原始数据文件 (用于兜底，如果Q1数据缺失太多)
RAW_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'

# 参数假设
HOURS_MORNING = 12.0  # 过夜暴露 (8pm - 8am)
HOURS_EVENING = 4.0  # 晚间暴露 (4pm - 8pm)
RAT_REDUCTION_ELASTICITY = 0.5  # 弹性系数


# ================= 1. 数据加载 (修复版) =================
def load_data():
    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        print(f"正在读取 {INPUT_FILE}...")

        # --- 关键修复：反推 Monthly_Trash_Tons ---
        if 'Monthly_Trash_Tons' not in df.columns:
            print("⚠️ 未找到 Monthly_Trash_Tons，正在根据 Tons_Per_Pickup * Freq 反推...")
            # 公式：月总量 = 单次量 * 每周频次 * 4.33周
            df['Monthly_Trash_Tons'] = df['Tons_Per_Pickup'] * df['Freq'] * 4.33

    else:
        print(f"⚠️ {INPUT_FILE} 不存在，尝试使用原始数据 {RAW_FILE}")
        df = pd.read_csv(RAW_FILE)
        df['Freq'] = 2  # 默认值

    # 确保没有空值干扰 (现在 Monthly_Trash_Tons 肯定有了)
    return df.dropna(subset=['Rat_Complaints', 'Monthly_Trash_Tons'])


# ================= 2. 相关性分析 =================
def analyze_correlation(df):
    print("\n=== Part 1: Trash vs Rats Correlation ===")

    # 尝试计算密度，如果没有面积数据就用绝对值
    if 'SHAPE_Area' in df.columns:
        df['Area_sqkm'] = df['SHAPE_Area'] * 9.2903e-8
        df['Trash_Density'] = df['Monthly_Trash_Tons'] / df['Area_sqkm']
        df['Rat_Density'] = df['Rat_Complaints'] / df['Area_sqkm']
        x_col, y_col = 'Trash_Density', 'Rat_Density'
        label = 'Density (per km²)'
    else:
        # 如果没有面积数据，直接用绝对量
        x_col, y_col = 'Monthly_Trash_Tons', 'Rat_Complaints'
        label = 'Absolute Count'

    # 计算相关性
    corr, p_val = pearsonr(df[x_col], df[y_col])
    print(f"Correlation ({x_col} vs {y_col}): {corr:.3f} (p={p_val:.3e})")

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'red'})

    # 标注点
    if 'CD_ID' in df.columns:
        for i, row in df.iterrows():
            plt.text(row[x_col], row[y_col], f"MN{int(row['CD_ID']) % 100:02d}", fontsize=9)

    plt.title(f'Investigating the Source: Trash vs Rats ({label})', fontsize=14)
    plt.xlabel(f'Trash Generation ({label})')
    plt.ylabel(f'Rat Complaints ({label})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Viz_Q4_Correlation.png', dpi=300)
    print("📊 相关性图已保存: Viz_Q4_Correlation.png")

    return corr


# ================= 3. 制定 Morning vs Evening 策略 =================
def assign_pickup_schedule(df):
    print("\n=== Part 2: Morning vs Evening Assignment ===")

    # 策略：鼠患高于中位数的区域 -> Evening Pickup
    threshold = df['Rat_Complaints'].median()

    def get_time_slot(rats):
        if rats > threshold:
            return 'Evening', HOURS_EVENING  # 4小时暴露
        else:
            return 'Morning', HOURS_MORNING  # 12小时暴露

    df[['Pickup_Time', 'Exposure_Hours']] = df['Rat_Complaints'].apply(lambda x: pd.Series(get_time_slot(x)))

    print("调度分配结果:")
    print(df['Pickup_Time'].value_counts())

    return df


# ================= 4. 预测影响 (Impact Prediction) =================
def predict_rat_reduction(df):
    print("\n=== Part 3: Impact on Rat Population ===")

    # 假设以前全是 Morning (12h)
    baseline_exposure = HOURS_MORNING

    # 计算新的暴露比率
    df['Exposure_Ratio'] = df['Exposure_Hours'] / baseline_exposure

    # 减少因子 = 弹性系数 * (1 - 暴露比率)
    # 例如：暴露时间从12h变4h (Ratio=0.33)，减少因子 = 0.5 * (1 - 0.33) = 0.33 (减少33%)
    df['Reduction_Factor'] = RAT_REDUCTION_ELASTICITY * (1 - df['Exposure_Ratio'])

    df['Predicted_Rats'] = df['Rat_Complaints'] * (1 - df['Reduction_Factor'])

    total_current = df['Rat_Complaints'].sum()
    total_predicted = df['Predicted_Rats'].sum()
    reduction_pct = (total_current - total_predicted) / total_current

    print(f"当前老鼠投诉总量: {int(total_current)}")
    print(f"预测老鼠投诉总量: {int(total_predicted)}")
    print(f"预计改善幅度: -{reduction_pct:.1%}")

    # 绘图：Before vs After
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    # 排序以便展示
    df_sorted = df.sort_values('Rat_Complaints', ascending=False)

    plt.bar(x - width / 2, df_sorted['Rat_Complaints'], width, label='Current (Baseline)', color='gray', alpha=0.7)
    plt.bar(x + width / 2, df_sorted['Predicted_Rats'], width, label='Predicted (After Strategy)', color='green',
            alpha=0.8)

    # 标记改成 Evening 的区
    evening_indices = [i for i, time in enumerate(df_sorted['Pickup_Time']) if time == 'Evening']
    # 只在这些柱子上画标记
    if evening_indices:
        plt.plot(evening_indices, [df_sorted['Rat_Complaints'].iloc[i] + 50 for i in evening_indices],
                 'v', color='orange', markersize=10, label='Switched to Evening Pickup', linestyle='None')

    # 处理 X 轴标签
    labels = [f"MN{int(cd) % 100:02d}" for cd in df_sorted['CD_ID']] if 'CD_ID' in df.columns else df_sorted.index
    plt.xticks(x, labels, rotation=45)

    plt.xlabel('Sanitation District')
    plt.ylabel('Rat Complaints Count')
    plt.title(f'Projected Impact of "Evening Pickup" Strategy (Total Reduction: {reduction_pct:.1%})')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('Viz_Q4_Impact_Prediction.png', dpi=300)
    print("📊 预测对比图已保存: Viz_Q4_Impact_Prediction.png")

    return df


# ================= 主程序 =================
if __name__ == "__main__":
    df = load_data()

    # 确保数据非空再继续
    if df is not None and not df.empty:
        corr = analyze_correlation(df)
        df = assign_pickup_schedule(df)
        df = predict_rat_reduction(df)

        # 导出结果
        df.to_csv('problem4_strategy_result.csv', index=False)
        print("\n✅ Q4 分析完成，策略表已保存至 problem4_strategy_result.csv")
    else:
        print("❌ 数据加载失败，无法进行分析。")