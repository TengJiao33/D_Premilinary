import pandas as pd
import numpy as np
import os
import math
import itertools
import random
import networkx as nx 

# ================= 1. 核心配置 =================
DATA_PATH = os.path.join('..', 'extra_data', 'merged_data', 'Manhattan_Data_Current_2023_2025.csv')
OUTPUT_FILE = 'problem1_final_solution.csv'

# 车辆参数 [cite: 88]
# 标准容量12吨，设定90%装载率以模拟路径损耗
TRUCK_CAPACITY = 12.0 * 1.0

# 物理与卫生约束 [cite: 90, 93]
MAX_GAP_DAYS = 4         # 允许最大间隔 (例如周一收完，下一次周五，间隔3天，Gap=4)
STREET_CAPACITY_TONS = 600.0 # 街道物理堆放上限 (假设值，可根据实际调整)

# === 权重配置 (调优后的魔法数字) ===
W_TRUCKS = 5000.0    # [最重要] 极力压低最大车队数 (CapEx)
W_VAR = 50.0         # 压低每日波动 (OpEx)
W_COHESION = 300.0   # [复活的关键] 奖励邻居同一天工作，引导"拼车"机会

# === 真实的曼哈顿拓扑结构 ===
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
# 构建图对象，用于计算连通分量
G = nx.Graph(REAL_TOPOLOGY)

# ================= 2. 数据加载与预处理 =================

def load_data(filepath):
    """加载数据并计算风险等级"""
    if not os.path.exists(filepath): 
        print("Warning: File not found. Using Mock Data.")
        return _generate_mock_data()

    df = pd.read_csv(filepath)
    districts = []
    
    # [cite: 101] 鼠患分析：如果投诉量高，必须高频清运
    rat_threshold = df['Rat_Complaints'].quantile(0.70) # 前30%为高风险
    
    for _, row in df.iterrows():
        cd_id = int(row['CD_ID']) if 'CD_ID' in row else int(row.name)
        name = f"MN{cd_id % 100:02d}"
        
        # 将月度数据转换为日均数据 [cite: 72]
        daily_tons = row['Monthly_Trash_Tons'] / 30.0
        rat_count = row['Rat_Complaints']
        
        # 判定风险等级
        is_high_risk = (rat_count >= rat_threshold)
        
        districts.append({
            'id': name,
            'daily_tons': daily_tons,
            'rats': rat_count,
            'is_high_risk': is_high_risk
        })
    return districts

def _generate_mock_data():
    """生成测试数据"""
    return [{'id': f'MN{i:02d}', 'daily_tons': 15.0 + i, 'rats': 100+i*10, 'is_high_risk': i%2==0} for i in range(1,13)]

def get_valid_patterns(district):
    """基于风险和物理约束，生成合法的排班模式"""
    daily_tons = district['daily_tons']
    must_be_frequent = district['is_high_risk']
    
    valid_patterns = []
    
    # 遍历 7 天的所有 0/1 组合
    for p in itertools.product([0, 1], repeat=7):
        freq = sum(p)
        
        # [cite: 90] 频率只能是 2 或 3
        if freq not in [2, 3]: continue
            
        # [cite: 101] 鼠患约束：高风险区必须 >= 3次
        if must_be_frequent and freq < 3: continue
            
        # 间隔计算
        pickup_days = [i for i, x in enumerate(p) if x == 1]
        gaps = []
        for i in range(len(pickup_days) - 1):
            gaps.append(pickup_days[i+1] - pickup_days[i])
        gaps.append((7 - pickup_days[-1]) + pickup_days[0])
        max_gap = max(gaps)
        
        # 卫生与物理爆仓死线
        if max_gap > MAX_GAP_DAYS: continue
        if (max_gap * daily_tons) > STREET_CAPACITY_TONS: continue 
            
        valid_patterns.append(np.array(p))
    
    # 保底逻辑：如果太严格导致没模式可选，强制给个 1010100
    if not valid_patterns:
        valid_patterns.append(np.array([1,0,1,0,1,0,0]))
        
    return valid_patterns

# ================= 3. 核心评估函数 (Topology Logic) =================

def calculate_trucks_with_topology(day_active_districts, district_map):
    """
    [cite: 95] 核心逻辑：基于拓扑的拼车计算
    只有连通的邻居才能共享卡车容量
    """
    if not day_active_districts: return 0
    
    # 找出当天的连通分量 (Connected Components)
    subgraph = G.subgraph(day_active_districts)
    components = list(nx.connected_components(subgraph))
    
    total_trucks_needed = 0
    
    for component in components:
        # 这一组邻居的总垃圾量
        component_total_load = sum(district_map[node]['load_today'] for node in component)
        # 这一组需要的车 (拼单后的向上取整)
        trucks = math.ceil(component_total_load / TRUCK_CAPACITY)
        total_trucks_needed += trucks
        
    return total_trucks_needed

def evaluate_solution(districts, indices):
    d_map = {d['id']: d for d in districts}
    daily_trucks = np.zeros(7)
    total_cohesion_score = 0
    
    # 1. 预计算每个区当前的单次清运量
    for i, d in enumerate(districts):
        pat = d['patterns'][indices[i]]
        d['current_pattern'] = pat
        # 假设均匀产生：单次量 = 日产量 * 7 / 频率
        d['pickup_load'] = d['daily_tons'] * 7.0 / sum(pat)
        
    # 2. 逐日计算
    for day in range(7):
        active_nodes = []
        for d in districts:
            if d['current_pattern'][day] == 1:
                active_nodes.append(d['id'])
                d_map[d['id']]['load_today'] = d['pickup_load']
        
        # A. 计算卡车需求 (Hard Cost)
        daily_trucks[day] = calculate_trucks_with_topology(active_nodes, d_map)
        
        # B. 计算内聚性 (Soft Reward) - 这是引导算法走出局部最优的关键！
        if len(active_nodes) > 1:
            subgraph = G.subgraph(active_nodes)
            # 边数越多，说明邻居同步率越高，越容易产生"拼车"
            total_cohesion_score += subgraph.number_of_edges()
            
    max_trucks = np.max(daily_trucks)
    var_trucks = np.var(daily_trucks)
    
    # Cost = (卡车数权重) + (波动权重) - (内聚奖励)
    cost = (W_TRUCKS * max_trucks) + (W_VAR * var_trucks) - (W_COHESION * total_cohesion_score)
    
    return cost, daily_trucks

# ================= 4. 模拟退火求解器 =================

def solve_sa(districts):
    # 初始化模式池
    print("正在生成合法模式池 (考虑鼠患风险 & 物理容量)...")
    for d in districts:
        d['patterns'] = get_valid_patterns(d)

    # 初始解
    current_idx = [random.randint(0, len(d['patterns'])-1) for d in districts]
    curr_cost, _ = evaluate_solution(districts, current_idx)
    best_cost = curr_cost
    best_idx = list(current_idx)
    
    # SA 参数
    T = 3000.0
    alpha = 0.99
    
    print(f"开始优化 (Initial Cost: {curr_cost:.1f})...")
    while T > 0.1:
        # 随机选择一个街区改变排班
        idx = random.randint(0, len(districts)-1)
        if len(districts[idx]['patterns']) <= 1: continue
        
        old_val = current_idx[idx]
        new_val = random.randint(0, len(districts[idx]['patterns'])-1)
        
        current_idx[idx] = new_val
        new_cost, _ = evaluate_solution(districts, current_idx)
        
        # Metropolis 准则
        delta = new_cost - curr_cost
        if delta < 0 or random.random() < math.exp(-delta/T):
            curr_cost = new_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_idx = list(current_idx)
        else:
            current_idx[idx] = old_val # Revert
            
        T *= alpha
        
    return districts, best_idx

# ================= 5. 结果分析与保存 =================

def analyze_and_save(districts, indices):
    """
    [cite: 95] 回答核心问题：共享到底省了多少车？
    """
    _, final_loads = evaluate_solution(districts, indices)
    
    # 计算 "孤岛模式" (No Sharing) 的需求
    no_share_max_trucks = 0
    days_no_share = np.zeros(7)
    
    for day in range(7):
        day_sum = 0
        for i, d in enumerate(districts):
            pat = d['patterns'][indices[i]]
            if pat[day] == 1:
                load = d['daily_tons'] * 7.0 / sum(pat)
                # 每个人单独派车
                day_sum += math.ceil(load / TRUCK_CAPACITY)
        days_no_share[day] = day_sum
    
    max_no_share = np.max(days_no_share)
    max_with_share = np.max(final_loads)
    saved = max_no_share - max_with_share
    
    print("\n" + "="*50)
    print(" 拓扑共享效益报告 (Problem D Answer) ")
    print("="*50)
    print(f"如果不共享 (孤岛模式) 需最大车队: {int(max_no_share)} 辆")
    print(f"基于相邻拓扑共享 (本方案) 需最大车队: {int(max_with_share)} 辆")
    print(f"节省车辆数: {int(saved)} 辆")
    print(f"效率提升: {(saved/max_no_share)*100:.2f}%")
    print("-" * 50)
    print("方案亮点:")
    print(f"1. 鼠患高发区已强制设置为 3次/周 (Risk Compliant)")
    print(f"2. 利用拓扑聚合减少了碎片空载 (Self-Consistent)")
    print("="*50)

    # 保存 CSV
    res = []
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for i, d in enumerate(districts):
        pat = d['patterns'][indices[i]]
        row = {
            'District': d['id'], 
            'Risk_Level': 'HIGH' if d['is_high_risk'] else 'Normal',
            'Avg_Daily_Tons': round(d['daily_tons'], 1),
            'Freq': sum(pat)
        }
        for j, val in enumerate(pat):
            row[days[j]] = '✓' if val else '-'
        res.append(row)
    
    df_out = pd.DataFrame(res)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"详细排班表已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    d_solved, idx_solved = solve_sa(data)
    analyze_and_save(d_solved, idx_solved)