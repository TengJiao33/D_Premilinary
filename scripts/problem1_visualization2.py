import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely import wkt
from shapely.geometry import Polygon, LineString
import networkx as nx
import numpy as np
import os

# ================= 配置区域 =================
# 地图数据路径 (请确保路径正确)
MAP_FILE = '../raw_data/DSNY_Districts_20251130.csv' 
# 排班结果路径 (上一轮生成的)
SCHEDULE_FILE = 'problem1_final_solution.csv'

# 真实的曼哈顿拓扑
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

# ================= 1. 数据加载模块 =================

def safe_wkt_load(wkt_string):
    try: return wkt.loads(wkt_string)
    except: return None

def create_mock_map():
    """如果找不到地图文件，创建一个简易的方格地图用于演示"""
    print("⚠️ 未找到地图文件，生成简易 Mock 地图...")
    polys = []
    # 模拟一个 2x6 的长条形曼哈顿
    ids = ['MN01','MN02','MN03','MN04','MN05','MN06',
           'MN07','MN08','MN09','MN10','MN11','MN12']
    coords = [
        (0,0), (1,0), (0,1), (1,1), (1,2), (0,2),
        (1,3), (0,3), (1,4), (0,4), (1,5), (0,5)
    ]
    
    for district_id, (x, y) in zip(ids, coords):
        # 创建一个 0.8 x 0.8 的方块
        poly = Polygon([(x, y), (x+0.9, y), (x+0.9, y+0.9), (x, y+0.9)])
        polys.append({'DISTRICT': district_id, 'geometry': poly})
        
    return gpd.GeoDataFrame(polys)

def load_data():
    # 1. 加载地图
    if os.path.exists(MAP_FILE):
        try:
            df = pd.read_csv(MAP_FILE)
            # 解析几何列
            if 'multipolygon' in df.columns:
                df['geometry'] = df['multipolygon'].apply(safe_wkt_load)
            elif 'geometry' in df.columns:
                df['geometry'] = df['geometry'].apply(safe_wkt_load)
                
            gdf = gpd.GeoDataFrame(df[df['geometry'].notna()], geometry='geometry')
            # 过滤曼哈顿
            gdf = gdf[gdf['DISTRICT'].str.startswith('MN')]
            print(f"✅ 成功加载地图文件: {len(gdf)} 个分区")
        except Exception as e:
            print(f"❌ 地图加载失败: {e}")
            gdf = create_mock_map()
    else:
        gdf = create_mock_map()
        
    # 2. 加载排班
    if os.path.exists(SCHEDULE_FILE):
        sched_df = pd.read_csv(SCHEDULE_FILE)
        print("✅ 成功加载排班表")
    else:
        print("⚠️ 未找到排班表，生成随机排班...")
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        data = []
        for d in gdf['DISTRICT'].unique():
            row = {'District': d}
            for day in days: row[day] = 'Pickup' if np.random.rand() > 0.5 else '-'
            data.append(row)
        sched_df = pd.DataFrame(data)

    return gdf, sched_df

# ================= 2. 核心计算逻辑 =================

def get_daily_clusters(active_districts):
    """
    计算当天的连通分量 (Sharing Groups)
    """
    if not active_districts: return []
    
    G = nx.Graph()
    G.add_nodes_from(active_districts)
    
    # 仅添加存在的边（即相邻关系）
    for node in active_districts:
        neighbors = REAL_TOPOLOGY.get(node, [])
        for nb in neighbors:
            if nb in active_districts:
                G.add_edge(node, nb)
    
    return list(nx.connected_components(G))

# ================= 3. 绘图逻辑 (已修复报错) =================

def plot_logistics_analysis(gdf, sched_df):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # 设置画布：2行4列 (最后一张画图例)
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    # 颜色池
    cluster_colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c']
    
    for i, day in enumerate(days):
        ax = axes[i]
        
        # 1. 绘制底图 (灰色背景)
        gdf.plot(ax=ax, color='#ecf0f1', edgecolor='white')
        
        # 2. 获取当天工作的区域
        day_col = sched_df[day].astype(str)
        active_mask = day_col.str.contains('Pickup') | day_col.str.contains('✓')
        active_districts = sched_df[active_mask]['District'].tolist()
        
        if not active_districts:
            ax.set_title(f"{day} (No Service)", fontsize=14)
            ax.axis('off')
            continue
            
        # 3. 计算连通分量
        clusters = get_daily_clusters(active_districts)
        
        # 4. 按 Cluster 染色并连线
        for c_idx, cluster in enumerate(clusters):
            color = cluster_colors[c_idx % len(cluster_colors)]
            
            # 染色
            cluster_gdf = gdf[gdf['DISTRICT'].isin(cluster)]
            cluster_gdf.plot(ax=ax, color=color, alpha=0.8, edgecolor='black')
            
            # 画内部连接线 (Topology Edges)
            if len(cluster) > 1:
                # === 修复核心: 安全构建坐标字典 ===
                c_dict = {}
                for idx, row in cluster_gdf.iterrows():
                    centroid = row['geometry'].centroid
                    c_dict[row['DISTRICT']] = (centroid.x, centroid.y)
                # =================================
                
                processed_edges = set()
                for node in cluster:
                    neighbors = REAL_TOPOLOGY.get(node, [])
                    for nb in neighbors:
                        if nb in cluster and tuple(sorted((node, nb))) not in processed_edges:
                            if node in c_dict and nb in c_dict:
                                p1 = c_dict[node]
                                p2 = c_dict[nb]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', linewidth=2, alpha=0.6, linestyle='-')
                                processed_edges.add(tuple(sorted((node, nb))))

        # 5. 标注不可行性 (Infeasibility Highlight)
        if len(clusters) > 1:
            try:
                c1_node = list(clusters[0])[0]
                c2_node = list(clusters[1])[0]
                
                # 获取坐标 (安全方式)
                p1_geo = gdf[gdf['DISTRICT']==c1_node].geometry.centroid.iloc[0]
                p2_geo = gdf[gdf['DISTRICT']==c2_node].geometry.centroid.iloc[0]
                
                # 画虚线
                ax.plot([p1_geo.x, p2_geo.x], [p1_geo.y, p2_geo.y], color='#e74c3c', linestyle=':', linewidth=2)
                # 画个叉
                mid_x, mid_y = (p1_geo.x + p2_geo.x)/2, (p1_geo.y + p2_geo.y)/2
                ax.text(mid_x, mid_y, "✘", color='red', fontsize=20, ha='center', va='center', fontweight='bold')
                ax.text(mid_x, mid_y-0.01, "No Sharing", color='red', fontsize=8, ha='center')
            except Exception as e:
                pass # 如果算不出坐标就跳过标注

        # 标注名字
        for _, row in gdf.iterrows():
            if row['DISTRICT'] in active_districts:
                try:
                    cent = row['geometry'].centroid
                    ax.annotate(row['DISTRICT'], (cent.x, cent.y), ha='center', fontsize=8, fontweight='bold', color='black')
                except: pass

        ax.set_title(f"{day}: {len(clusters)} Groups", fontsize=14, fontweight='bold')
        ax.axis('off')

    # 最后一个子图画图例
    ax_legend = axes[7]
    ax_legend.axis('off')
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Group A (Sharing OK)'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Group B (Sharing OK)'),
        mpatches.Patch(facecolor='#ecf0f1', edgecolor='gray', label='Inactive District'),
        plt.Line2D([0], [0], color='white', lw=2, label='Shared Route'),
        plt.Line2D([0], [0], color='#e74c3c', lw=2, linestyle=':', label='Infeasible Link (✘)')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12, title="Logistics Topology")
    ax_legend.set_title("Why Global Pooling Fails?", fontsize=14, color='darkred')
    
    plt.tight_layout()
    output_png = 'Viz_Advanced_Infeasibility.png'
    plt.savefig(output_png, dpi=300)
    print(f"🖼️ 可视化生成完毕: {output_png}")
    plt.show()

if __name__ == "__main__":
    gdf, sched = load_data()
    plot_logistics_analysis(gdf, sched)