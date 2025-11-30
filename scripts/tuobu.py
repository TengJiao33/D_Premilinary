import pandas as pd
import geopandas as gpd
from shapely import wkt
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from networkx.algorithms import community

# --- 辅助函数保持不变 ---
def safe_wkt_load(wkt_string):
    try:
        return wkt.loads(wkt_string)
    except:
        return None

def plot_refined_topology():
    # --- 1. 数据读取与处理 (保持不变) ---
    csv_file_path = './raw_data/DSNY_Districts_20251130.csv'
    if not os.path.exists(csv_file_path):
        print("❌ 找不到 CSV 文件。")
        return

    print("📂 读取数据...")
    df = pd.read_csv(csv_file_path, usecols=['DISTRICT', 'SHAPE_Area', 'multipolygon'])
    df = df[df['DISTRICT'].str.startswith('MN', na=False)].copy()
    
    df['geometry'] = df['multipolygon'].apply(safe_wkt_load)
    gdf = gpd.GeoDataFrame(df, geometry='geometry').dropna(subset=['geometry'])
    
    gdf['Area_Float'] = gdf['SHAPE_Area'].astype(str).str.replace(',', '').astype(float)
    min_area = gdf['Area_Float'].min()
    max_area = gdf['Area_Float'].max()
    
    # 【调整点 3】整体增大节点尺寸，给文字腾地方
    # 原来是 1000-3000，现在改成 1500-4000
    gdf['node_size'] = 1500 + (gdf['Area_Float'] - min_area) / (max_area - min_area) * 2500

    # --- 2. 构建图网络 (保持不变) ---
    G = nx.Graph()
    for idx, row in gdf.iterrows():
        G.add_node(row['DISTRICT'], size=row['node_size'])

    gdf['geometry'] = gdf['geometry'].buffer(0)
    for i, row_i in gdf.iterrows():
        for j, row_j in gdf.iterrows():
            if i >= j: continue
            if row_i['geometry'].distance(row_j['geometry']) < 1e-3:
                G.add_edge(row_i['DISTRICT'], row_j['DISTRICT'])

    isolates = list(nx.isolates(G))
    for iso in isolates:
        if iso == 'MN12':
            G.add_edge('MN12', 'MN11') 
            G.add_edge('MN12', 'MN09')
    
    # --- 3. 社区检测与配色 (解释颜色的来源) ---
    communities = list(community.greedy_modularity_communities(G))
    color_map = {}
    # 选用一套更专业、对比度更强的配色方案
    palette = ['#E63946', '#457B9D', '#F4A261', '#2A9D8F'] 
    
    for i, comm in enumerate(communities):
        c_color = palette[i % len(palette)]
        for node in comm:
            color_map[node] = c_color

    node_colors = [color_map.get(n, '#CCCCCC') for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

    # --- 4. 布局与绘图 (重点修改) ---
    plt.figure(figsize=(12, 10))
    
    # 布局算法 (固定种子，保证结果一致)
    pos = nx.spring_layout(G, k=0.5, seed=42, iterations=100)
    
    # 画边
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.4, edge_color='#888888')
    
    # 画节点
    nx.draw_networkx_nodes(G, pos, 
                           node_size=node_sizes, 
                           node_color=node_colors, 
                           edgecolors='#333333', # 边框颜色深一点
                           linewidths=1.5)
    
    # 【调整点 3】画标签：缩小字号，改用白色字体对比度更高
    nx.draw_networkx_labels(G, pos, 
                            font_size=9, # 字号调小 (原来是11)
                            font_weight='bold', 
                            font_color='white', # 白色字在深色背景上更清晰
                            font_family='sans-serif')

    # --- 图例修改 (Legend Refinement) ---
    legend_patches = []
    for i in range(len(communities)):
        label = f"Sharing Zone {i+1} (Group {i+1})"
        patch = mpatches.Patch(color=palette[i % len(palette)], label=label)
        legend_patches.append(patch)
        
    # 【调整点 1】图例位置移到左上角 (loc='upper left')
    # 加上 bbox_to_anchor 微调位置，让它离边缘远点
    plt.legend(handles=legend_patches, 
               loc='upper left', 
               bbox_to_anchor=(0.02, 0.98),
               fontsize=11, 
               frameon=True, 
               shadow=False, # 去掉阴影，更清爽
               facecolor='white',
               edgecolor='#CCCCCC')

    plt.title("Optimized Waste Sharing Network (Abstract Topology)", fontsize=16, fontweight='bold', pad=15)
    plt.axis('off')
    
    output_file = 'refined_topology.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 精修版拓扑图已生成: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_refined_topology()