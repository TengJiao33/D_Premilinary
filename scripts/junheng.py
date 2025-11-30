import pandas as pd
import numpy as np
import os
import math
import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt

# ================= 1. 核心配置与数据结构 =================
DATA_PATH = os.path.join('..', 'extra_data', 'merged_data', 'Manhattan_Data_Current_2023_2025.csv')

# 车辆参数
TRUCK_CAPACITY = 12.0 * 0.9  
MAX_GAP_DAYS = 4         
STREET_CAPACITY_TONS = 600.0 

# 初始权重 (实验中会修改它)
W_TRUCKS = 5000.0    
W_VAR_DEFAULT = 50.0  # 默认的均衡权重
W_COHESION = 300.0   

# 拓扑结构
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
G = nx.Graph(REAL_TOPOLOGY)

# 全局变量，用于动态修改权重
CURRENT_W_VAR = W_VAR_DEFAULT

# ================= 2. 求解器核心函数 =================

def load_data(filepath):
    if not os.path.exists(filepath): 
        # Mock data for testing without file
        return [{'id': f'MN{i:02d}', 'daily_tons': 15.0+i, 'rats': 100+i*10, 'is_high_risk': i%2==0} for i in range(1,13)]

    df = pd.read_csv(filepath)
    districts = []
    rat_threshold = df['Rat_Complaints'].quantile(0.70)
    
    for _, row in df.iterrows():
        cd_id = int(row['CD_ID']) if 'CD_ID' in row else int(row.name)
        name = f"MN{cd_id % 100:02d}"
        daily_tons = row['Monthly_Trash_Tons'] / 30.0
        rat_count = row['Rat_Complaints']
        districts.append({
            'id': name,
            'daily_tons': daily_tons,
            'rats': rat_count,
            'is_high_risk': (rat_count >= rat_threshold)
        })
    return districts

def get_valid_patterns(district):
    daily_tons = district['daily_tons']
    must_be_frequent = district['is_high_risk']
    valid_patterns = []
    
    for p in itertools.product([0, 1], repeat=7):
        freq = sum(p)
        if freq not in [2, 3]: continue
        if must_be_frequent and freq < 3: continue
            
        pickup_days = [i for i, x in enumerate(p) if x == 1]
        gaps = [pickup_days[i+1]-pickup_days[i] for i in range(len(pickup_days)-1)]
        gaps.append((7 - pickup_days[-1]) + pickup_days[0])
        max_gap = max(gaps)
        
        if max_gap > MAX_GAP_DAYS: continue
        if (max_gap * daily_tons) > STREET_CAPACITY_TONS: continue 
            
        valid_patterns.append(np.array(p))
    
    if not valid_patterns: valid_patterns.append(np.array([1,0,1,0,1,0,0]))
    return valid_patterns

def calculate_trucks_with_topology(day_active_districts, district_map):
    if not day_active_districts: return 0
    subgraph = G.subgraph(day_active_districts)
    components = list(nx.connected_components(subgraph))
    total_trucks = 0
    for component in components:
        load = sum(district_map[node]['load_today'] for node in component)
        total_trucks += math.ceil(load / TRUCK_CAPACITY)
    return total_trucks

def evaluate_solution(districts, indices):
    d_map = {d['id']: d for d in districts}
    daily_trucks = np.zeros(7)
    total_cohesion_score = 0
    
    for i, d in enumerate(districts):
        pat = d['patterns'][indices[i]]
        d['current_pattern'] = pat
        d['pickup_load'] = d['daily_tons'] * 7.0 / sum(pat)
        
    for day in range(7):
        active_nodes = []
        for d in districts:
            if d['current_pattern'][day] == 1:
                active_nodes.append(d['id'])
                d_map[d['id']]['load_today'] = d['pickup_load']
        
        daily_trucks[day] = calculate_trucks_with_topology(active_nodes, d_map)
        
        if len(active_nodes) > 1:
            subgraph = G.subgraph(active_nodes)
            total_cohesion_score += subgraph.number_of_edges()
            
    max_trucks = np.max(daily_trucks)
    var_trucks = np.var(daily_trucks)
    
    # 使用动态全局变量 CURRENT_W_VAR
    cost = (W_TRUCKS * max_trucks) + (CURRENT_W_VAR * var_trucks) - (W_COHESION * total_cohesion_score)
    return cost, daily_trucks

def solve_sa(districts):
    # 预处理模式
    for d in districts:
        if 'patterns' not in d: d['patterns'] = get_valid_patterns(d)

    current_idx = [random.randint(0, len(d['patterns'])-1) for d in districts]
    curr_cost, _ = evaluate_solution(districts, current_idx)
    best_cost = curr_cost
    best_idx = list(current_idx)
    
    # 快速退火配置 (为了实验跑得快一点，步数设少一点，但足以看出趋势)
    T = 2000.0
    alpha = 0.98
    
    while T > 0.5:
        idx = random.randint(0, len(districts)-1)
        if len(districts[idx]['patterns']) <= 1: continue
        
        old_val = current_idx[idx]
        new_val = random.randint(0, len(districts[idx]['patterns'])-1)
        
        current_idx[idx] = new_val
        new_cost, _ = evaluate_solution(districts, current_idx)
        
        if new_cost < curr_cost or random.random() < math.exp(-(new_cost-curr_cost)/T):
            curr_cost = new_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_idx = list(current_idx)
        else:
            current_idx[idx] = old_val
        T *= alpha
    return districts, best_idx

# ================= 3. 实验逻辑 =================

def run_experiment(w_var_value, strategy_name):
    global CURRENT_W_VAR
    CURRENT_W_VAR = w_var_value # 修改全局权重
    
    print(f"\n🚀 开始实验: {strategy_name} (均衡惩罚权重 W_VAR={w_var_value}) ...")
    
    # 每次重新加载数据以清除状态
    data = load_data(DATA_PATH)
    
    # 运行求解器
    districts, indices = solve_sa(data)
    
    # 评估结果
    cost, daily_trucks = evaluate_solution(districts, indices)
    max_trucks = np.max(daily_trucks)
    total_truck_days = np.sum(daily_trucks)
    
    print(f"   -> 结果: 最大车队={int(max_trucks)}, 方差={np.var(daily_trucks):.2f}")
    return daily_trucks, max_trucks, total_truck_days

if __name__ == "__main__":
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)
    
    # 1. 运行三组对比实验
    # 方案 A: 你的当前方案 (追求均衡)
    d1, max1, tot1 = run_experiment(50.0, "Balanced (Proposed)")
    
    # 方案 B: 完全不管均衡 (只管拓扑拼车和总数)
    d2, max2, tot2 = run_experiment(0.0, "Unbalanced (No Penalty)")
    
    # 方案 C: 故意制造拥堵 (负权重)
    d3, max3, tot3 = run_experiment(-100.0, "Anti-Balanced (Chaos)")
    
    # 2. 打印详细对比表
    print("\n" + "="*80)
    print(f"{'STRATEGY':<25} | {'MAX TRUCKS (CapEx)':<20} | {'VARIANCE':<10} | {'TOTAL LOAD'}")
    print("-" * 80)
    print(f"{'Balanced (Our Model)':<25} | {int(max1):<20} | {np.var(d1):.1f}       | {int(tot1)}")
    print(f"{'No Balance Constraint':<25} | {int(max2):<20} | {np.var(d2):.1f}       | {int(tot2)}")
    print(f"{'Forced Imbalance':<25} | {int(max3):<20} | {np.var(d3):.1f}       | {int(tot3)}")
    print("="*80)
    
    # 3. 绘图证明
    plt.figure(figsize=(10, 6))
    days_label = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    
    plt.plot(days_label, d1, marker='o', linewidth=3, color='#2ecc71', label=f'Balanced (Our Model): Max={int(max1)}')
    plt.plot(days_label, d2, marker='x', linestyle='--', color='#e74c3c', label=f'No Balance: Max={int(max2)}')
    
    plt.title("Proof of Optimality", fontsize=14)
    plt.ylabel("Trucks Required")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "Proof_Experiment_Result.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ 证明图表已生成: {output_file}")
    plt.show()