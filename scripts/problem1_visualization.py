import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import math
import os

# ================= 配置 =================
INPUT_FILE = 'problem1_final_solution.csv'
# 设置绘图风格 - 学术风
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 车辆参数 (必须与建模代码一致)
TRUCK_CAPACITY = 12.0 * 0.9 

# 拓扑结构 (用于画网络图)
REAL_TOPOLOGY = {
    'MN01': ['MN02', 'MN03'],
    'MN02': ['MN01', 'MN03', 'MN04'],
    'MN03': ['MN01', 'MN02', 'MN06'],
    'MN04': ['MN02', 'MN05', 'MN07'],
    'MN05': ['MN04', 'MN06', 'MN07'],
    'MN06': ['MN03', 'MN05', 'MN08'],
    'MN07': ['MN04', 'MN05', 'MN08', 'MN09'],
    'MN08': ['MN06', 'MN07', 'MN11'],
    'MN09': ['MN07', 'MN10', 'MN12'],
    'MN10': ['MN09', 'MN11', 'MN12'],
    'MN11': ['MN08', 'MN10', 'MN12'],
    'MN12': ['MN09', 'MN10', 'MN11']
}

# ================= 数据加载与处理 =================

def load_or_mock_data():
    if os.path.exists(INPUT_FILE):
        print(f"Loading {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
    else:
        print("CSV not found. Generating Mock Data for visualization...")
        # 模拟数据结构
        data = []
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        for i in range(1, 13):
            is_high = i in [3, 8, 9, 12] # 假定高风险区
            row = {
                'District': f'MN{i:02d}',
                'Risk_Level': 'HIGH' if is_high else 'Normal',
                'Avg_Daily_Tons': np.random.uniform(10, 25),
                'Freq': 3 if is_high else 2
            }
            # 随机生成排班
            pat = np.random.choice([0, 1], size=7, p=[0.6, 0.4])
            if sum(pat) == 0: pat[0]=1; pat[3]=1
            for d_idx, d_name in enumerate(days):
                row[d_name] = '✓' if pat[d_idx] == 1 else '-'
            data.append(row)
        df = pd.DataFrame(data)
    
    # 转换排班符号为数字
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for day in days:
        df[f'{day}_Num'] = df[day].apply(lambda x: 1 if '✓' in str(x) or 'Pickup' in str(x) else 0)
    
    return df

def calculate_daily_trucks_with_topology(df):
    """重算每天的卡车需求（带拓扑逻辑）"""
    G = nx.Graph(REAL_TOPOLOGY)
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    daily_trucks = []
    
    for day in days:
        # 1. 找出当天工作的节点
        active_rows = df[df[f'{day}_Num'] == 1]
        active_nodes = active_rows['District'].tolist()
        
        if not active_nodes:
            daily_trucks.append(0)
            continue
            
        # 2. 计算当天每个区的单次运量
        # load = daily_tons * 7 / freq
        node_loads = {}
        for _, row in active_rows.iterrows():
            load = row['Avg_Daily_Tons'] * 7.0 / row['Freq']
            node_loads[row['District']] = load
            
        # 3. 拓扑聚类
        subgraph = G.subgraph(active_nodes)
        components = list(nx.connected_components(subgraph))
        
        total_trucks = 0
        for comp in components:
            comp_load = sum(node_loads[n] for n in comp)
            total_trucks += math.ceil(comp_load / TRUCK_CAPACITY)
        
        daily_trucks.append(total_trucks)
        
    return days, daily_trucks

# ================= 绘图函数 =================

def plot_schedule_heatmap(df):
    """图1：排班热力图 - 展示错峰情况"""
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    # 准备矩阵数据
    matrix = df[[f'{day}_Num' for day in days]].values
    yticklabels = df['District'] + " (" + df['Risk_Level'].str[0] + ")" # 显示风险等级首字母
    
    plt.figure(figsize=(10, 6))
    # 自定义颜色: 0=白色, 1=深绿色
    cmap = sns.color_palette(["#f7f7f7", "#2ecc71"]) 
    
    ax = sns.heatmap(matrix, cmap=cmap, linewidths=.5, linecolor='lightgray',
                     yticklabels=yticklabels, xticklabels=days, cbar=False)
    
    # 标记格子
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] == 1:
                ax.text(x + 0.5, y + 0.5, 'Pickup', 
                        color='white', ha='center', va='center', weight='bold', size=9)

    plt.title('Optimized Collection Schedule (H=High Risk)', fontsize=16, pad=20)
    plt.ylabel('District')
    plt.tight_layout()
    plt.savefig('Viz_1_Schedule_Heatmap.png', dpi=300)
    print("Generated: Viz_1_Schedule_Heatmap.png")

def plot_truck_demand(days, trucks):
    """图2：每日车队需求 - 展示均衡性"""
    plt.figure(figsize=(10, 5))
    
    # 绘制柱状图
    bars = plt.bar(days, trucks, color='#3498db', alpha=0.8, width=0.6, edgecolor='black')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # 添加平均线
    avg_trucks = np.mean(trucks)
    plt.axhline(y=avg_trucks, color='#e74c3c', linestyle='--', linewidth=2, label=f'Avg: {avg_trucks:.1f}')
    
    plt.title('Daily Fleet Requirement (Topology-Aware)', fontsize=16, pad=20)
    plt.ylabel('Number of Trucks Needed')
    plt.ylim(0, max(trucks)*1.2)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Viz_2_Daily_Trucks.png', dpi=300)
    print("Generated: Viz_2_Daily_Trucks.png")

def plot_topology_network(df, target_day='Mon'):
    """图3：拓扑网络图 - 展示拼车效应"""
    G = nx.Graph(REAL_TOPOLOGY)
    
    # 确定当天工作的节点
    active_mask = df[f'{target_day}_Num'] == 1
    active_nodes = df[active_mask]['District'].tolist()
    inactive_nodes = df[~active_mask]['District'].tolist()
    
    # 设置节点位置 (曼哈顿简易坐标模拟)
    # 这是一个简化的 2x6 网格模拟曼哈顿狭长地形
    pos = {
        'MN12': (0, 5), 'MN11': (1, 5),
        'MN10': (0, 4), 'MN09': (1, 4),
        'MN08': (0, 3), 'MN07': (1, 3),
        'MN06': (0, 2), 'MN05': (1, 2),
        'MN03': (0, 1), 'MN04': (1, 1),
        'MN01': (0, 0), 'MN02': (1, 0)
    }
    
    plt.figure(figsize=(8, 10))
    
    # 画背景边
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1.5, style='--')
    
    # 画不工作的点
    nx.draw_networkx_nodes(G, pos, nodelist=inactive_nodes, node_color='white', 
                           edgecolors='gray', node_size=1000, label='Inactive')
    nx.draw_networkx_labels(G, pos, labels={n:n for n in inactive_nodes}, font_color='gray')
    
    # 画工作的点
    # 只有相邻的工作点之间才画实线边 (表示拼车)
    subgraph = G.subgraph(active_nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=active_nodes, node_color='#2ecc71', 
                           edgecolors='black', node_size=1200, label='Active (Pickup)')
    nx.draw_networkx_edges(subgraph, pos, edge_color='#27ae60', width=4)
    nx.draw_networkx_labels(G, pos, labels={n:n for n in active_nodes}, font_color='white', font_weight='bold')

    plt.title(f'Topology Sharing Analysis ({target_day})', fontsize=16)
    plt.axis('off')
    # 手动添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=15, label='Active District'),
        Line2D([0], [0], color='#27ae60', lw=4, label='Resource Sharing Link'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='gray', markersize=10, label='No Service')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Viz_3_Topology_Net.png', dpi=300)
    print("Generated: Viz_3_Topology_Net.png")

def plot_risk_compliance(df):
    """图4：风险合规性检查"""
    plt.figure(figsize=(8, 6))
    
    # 统计 High Risk vs Normal Risk 的频率分布
    sns.countplot(data=df, x='Risk_Level', hue='Freq', palette='viridis')
    
    plt.title('Compliance Check: Risk Level vs Frequency', fontsize=16)
    plt.xlabel('District Risk Classification')
    plt.ylabel('Count of Districts')
    plt.legend(title='Weekly Frequency')
    
    # 添加注释
    plt.figtext(0.5, 0.01, "Note: High Risk districts MUST have Frequency=3 (Model Constraint)", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig('Viz_4_Risk_Compliance.png', dpi=300)
    print("Generated: Viz_4_Risk_Compliance.png")

# ================= 主程序 =================

if __name__ == "__main__":
    # 1. 加载数据
    df = load_or_mock_data()
    
    # 2. 计算每日卡车 (用于绘图)
    days, trucks = calculate_daily_trucks_with_topology(df)
    
    # 3. 生成图表
    print("Starting visualization generation...")
    
    # 图 1: 排班表
    plot_schedule_heatmap(df)
    
    # 图 2: 每日卡车均衡图
    plot_truck_demand(days, trucks)
    
    # 图 3: 拓扑拼车示意图 (以周一为例)
    plot_topology_network(df, target_day='Mon')
    
    # 图 4: 鼠患风险合规图
    plot_risk_compliance(df)
    
    print("\nVisualization Complete! Check the .png files in your folder.")