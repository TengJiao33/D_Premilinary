import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform

# ================= 配置 =================
DATA_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'
OUTPUT_DIR = '.'

def configure_fonts():
    """自动配置中文字体"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def load_and_process_data():
    if not os.path.exists(DATA_FILE):
        print("❌ 数据文件不存在，生成模拟数据演示...")
        # 生成模拟数据用于演示代码功能
        np.random.seed(42)
        n = 59
        return pd.DataFrame({
            'CD_ID': range(101, 101+n),
            'Monthly_Trash_Tons': np.random.normal(1500, 300, n),
            'Rat_Complaints': np.random.normal(100, 40, n) + np.random.normal(1500, 300, n)*0.05
        })
    
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=['Rat_Complaints', 'Monthly_Trash_Tons'])
    return df

def plot_all_charts():
    configure_fonts()
    df = load_and_process_data()
    
    # ---------------------------------------------------------
    # 图 1: 相关性散点图 (Motivation)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 使用 Seaborn 画带回归线的散点图
    sns.regplot(x='Monthly_Trash_Tons', y='Rat_Complaints', data=df, 
                scatter_kws={'alpha':0.6, 'color':'#2980b9'}, 
                line_kws={'color':'#c0392b', 'label':'线性拟合 (Linear Fit)'})
    
    # 标出一些极端点 (Top 3)
    top_rats = df.nlargest(3, 'Rat_Complaints')
    for _, row in top_rats.iterrows():
        plt.text(row['Monthly_Trash_Tons'], row['Rat_Complaints'], 
                 f" CD{int(row['CD_ID'])}", fontsize=9, color='black')

    plt.title('数据验证: 垃圾产量与鼠患投诉的相关性分析', fontsize=16, fontweight='bold')
    plt.xlabel('月均垃圾产量 (吨)', fontsize=12)
    plt.ylabel('鼠患投诉数量 (次)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig('Chart1_Correlation.png', dpi=300, bbox_inches='tight')
    print("✅ 图 1 (相关性) 已保存。")
    plt.close()

    # ---------------------------------------------------------
    # 图 2: 策略成效对比 (Impact Assessment)
    # ---------------------------------------------------------
    # 计算逻辑 (复用之前的 TEH 模型)
    rat_threshold = df['Rat_Complaints'].quantile(0.60) # Top 40%
    df['Strategy'] = df['Rat_Complaints'].apply(lambda x: 'AM (早班)' if x >= rat_threshold else 'PM (晚班)')
    
    # 计算 Baseline (全 PM) vs Optimized (AM/PM) 的暴露指数
    # 假设: 垃圾量 * 暴露小时数
    df['Daily_Tons'] = df['Monthly_Trash_Tons'] / 30.0
    baseline_teh = (df['Daily_Tons'] * 22.0).sum() # 假设现状全是晚班(22h暴露)
    
    # 优化后: 早班11h, 晚班22h
    optimized_teh = df.apply(
        lambda row: row['Daily_Tons'] * 11.0 if row['Strategy']=='AM (早班)' else row['Daily_Tons'] * 22.0, 
        axis=1
    ).sum()
    
    reduction = (baseline_teh - optimized_teh) / baseline_teh * 100
    
    # 绘图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['现状 (Baseline)', '策略优化后 (Optimized)'], 
                   [baseline_teh, optimized_teh], 
                   color=['#95a5a6', '#2ecc71'], 
                   width=0.5)
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}\nTEH',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 画下降箭头
    mid_x = (bars[0].get_x() + bars[1].get_x() + bars[0].get_width()) / 2
    plt.annotate(f'风险降低 {reduction:.1f}%', 
                 xy=(1, optimized_teh), xytext=(0.5, baseline_teh),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=14, color='red', fontweight='bold', ha='center')

    plt.title('策略成效: 垃圾滞留暴露指数 (TEH) 对比', fontsize=16, fontweight='bold')
    plt.ylabel('总暴露量 (吨·小时)', fontsize=12)
    plt.ylim(0, baseline_teh * 1.2) # 留出顶部空间
    
    plt.savefig('Chart2_Impact.png', dpi=300, bbox_inches='tight')
    print("✅ 图 2 (成效对比) 已保存。")
    plt.close()

    # ---------------------------------------------------------
    # 图 3: 帕累托累积图 (Pareto - Justification)
    # ---------------------------------------------------------
    # 目的: 证明为什么要选 Top 40% 的区域
    df_sorted = df.sort_values(by='Rat_Complaints', ascending=False).reset_index(drop=True)
    df_sorted['Cumulative_Rats'] = df_sorted['Rat_Complaints'].cumsum()
    df_sorted['Cumulative_Pct'] = df_sorted['Cumulative_Rats'] / df_sorted['Rat_Complaints'].sum() * 100
    df_sorted['District_Pct'] = (df_sorted.index + 1) / len(df_sorted) * 100

    plt.figure(figsize=(10, 6))
    
    # 画线
    plt.plot(df_sorted['District_Pct'], df_sorted['Cumulative_Pct'], 
             color='#8e44ad', linewidth=3, label='鼠患累积占比')
    
    # 画 40% 切割线
    cut_x = 40
    cut_y = df_sorted[df_sorted['District_Pct'] >= 40]['Cumulative_Pct'].iloc[0]
    
    plt.axvline(x=cut_x, color='red', linestyle='--', alpha=0.6)
    plt.axhline(y=cut_y, color='red', linestyle='--', alpha=0.6)
    
    plt.scatter([cut_x], [cut_y], color='red', s=100, zorder=5)
    plt.text(cut_x + 2, cut_y - 5, 
             f'阈值点 (Threshold):\n前 {cut_x}% 的区域\n贡献了 {cut_y:.1f}% 的鼠患', 
             color='red', fontsize=11, fontweight='bold')

    # 填充颜色
    plt.fill_between(df_sorted['District_Pct'], 0, df_sorted['Cumulative_Pct'], 
                     where=(df_sorted['District_Pct'] <= cut_x),
                     color='#f1c40f', alpha=0.3, label='AM 早班覆盖区 (High Risk)')
    
    plt.title('分级策略依据: 鼠患分布的帕累托效应', fontsize=16, fontweight='bold')
    plt.xlabel('社区数量累积百分比 (%)', fontsize=12)
    plt.ylabel('鼠患投诉累积百分比 (%)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('Chart3_Pareto.png', dpi=300, bbox_inches='tight')
    print("✅ 图 3 (帕累托) 已保存。")
    plt.close()

if __name__ == "__main__":
    plot_all_charts()