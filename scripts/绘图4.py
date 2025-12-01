import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import platform

# ================= 1. 基础配置与字体设置 =================
# 地图数据路径 (请确认你的文件名)
MAP_FILE = './raw_data/DSNY_Districts_20251130.csv'
# 分析数据路径 (L5模型输出)
DATA_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'

def configure_chinese_font():
    """自动配置中文字体，防止乱码"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin': # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else: # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    print("✅ 中文字体配置完成")

# ================= 2. 数据加载函数 (保持逻辑稳健) =================
def safe_wkt_load(wkt_string):
    try: return wkt.loads(wkt_string)
    except: return None

def load_data():
    # 1. 加载地图
    if not os.path.exists(MAP_FILE):
        print(f"❌ 找不到地图文件: {MAP_FILE}")
        return None
    
    df_map = pd.read_csv(MAP_FILE)
    # 筛选曼哈顿 (MN开头)
    df_map = df_map[df_map['DISTRICT'].str.startswith('MN', na=False)].copy()
    df_map['geometry'] = df_map['multipolygon'].apply(safe_wkt_load)
    gdf_map = gpd.GeoDataFrame(df_map.dropna(subset=['geometry']), geometry='geometry')
    
    # 2. 加载分析数据
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到数据文件: {DATA_FILE}")
        return None
        
    df_data = pd.read_csv(DATA_FILE)
    
    # 3. 计算 Problem 4 策略 (AM/PM)
    # 逻辑: 鼠患最严重的 Top 40% -> 早班 (AM)
    rat_threshold = df_data['Rat_Complaints'].quantile(0.60)
    
    def get_shift_label(rats):
        if rats >= rat_threshold:
            return '早班 (AM) - 高风险'
        else:
            return '晚班 (PM) - 低风险'
            
    df_data['Shift_Label'] = df_data['Rat_Complaints'].apply(get_shift_label)
    
    # 4. 统一 ID 格式 (101 -> MN01) 以便合并
    def convert_id(cd_id):
        try:
            return f"MN{int(cd_id)%100:02d}"
        except:
            return str(cd_id)
    df_data['DISTRICT'] = df_data['CD_ID'].apply(convert_id)
    
    # 5. 合并数据
    merged = gdf_map.merge(df_data[['DISTRICT', 'Rat_Complaints', 'Shift_Label']], on='DISTRICT', how='left')
    return merged

# ================= 3. 绘图主程序 =================
def plot_charts():
    configure_chinese_font() # 设置字体
    
    gdf = load_data()
    if gdf is None: return

    # --- 图 1: 鼠患风险热力图 ---
    print("正在绘制图 1: 鼠患热力图...")
    fig1, ax1 = plt.subplots(figsize=(10, 12))
    
    gdf.plot(column='Rat_Complaints', 
             cmap='Reds',      # 红色系
             linewidth=0.8, 
             edgecolor='0.5', 
             legend=True,
             legend_kwds={'label': "鼠患投诉数量 (2023-2025)", 'orientation': "horizontal", 'shrink': 0.8},
             ax=ax1)
    
    ax1.set_title("曼哈顿鼠患风险分布现状 (Baseline)", fontsize=18, fontweight='bold')
    ax1.axis('off')
    
    # 标注 ID
    for _, row in gdf.iterrows():
        if row['geometry']:
            centroid = row['geometry'].centroid
            ax1.annotate(text=row['DISTRICT'], xy=(centroid.x, centroid.y), 
                         ha='center', fontsize=8, color='black', alpha=0.7)
    
    output1 = 'Rat_Risk_Map.png'
    plt.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"🖼️ 图 1 已保存: {output1}")
    plt.close(fig1) # 释放内存

    # --- 图 2: 早晚班战略部署图 ---
    print("正在绘制图 2: 战略部署图...")
    fig2, ax2 = plt.subplots(figsize=(10, 12))
    
    # 定义颜色: 早班(亮黄), 晚班(深蓝)
    color_map = {
        '早班 (AM) - 高风险': '#F1C40F', 
        '晚班 (PM) - 低风险': '#2C3E50'
    }
    
    # 分类绘图
    for label, color in color_map.items():
        subset = gdf[gdf['Shift_Label'] == label]
        if not subset.empty:
            subset.plot(ax=ax2, color=color, edgecolor='white', linewidth=1.0)
            
    # 处理缺失值 (如果有)
    missing = gdf[gdf['Shift_Label'].isna()]
    if not missing.empty:
        missing.plot(ax=ax2, color='lightgrey', hatch='///', edgecolor='white')

    ax2.set_title("Problem 4: 垃圾清运早晚班战略部署", fontsize=18, fontweight='bold')
    ax2.axis('off')
    
    # 自定义图例 (中文)
    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax2.legend(handles=patches, loc='upper left', fontsize=12, frameon=True, framealpha=0.9)

    # 标注 ID (白色字体更清晰)
    for _, row in gdf.iterrows():
        if row['geometry']:
            centroid = row['geometry'].centroid
            # 晚班区域背景深，用白色字；早班用黑色字
            text_color = 'white' if '晚班' in str(row['Shift_Label']) else 'black'
            ax2.annotate(text=row['DISTRICT'], xy=(centroid.x, centroid.y), 
                         ha='center', fontsize=9, color=text_color, fontweight='bold')

    output2 = 'Strategy_Shift_Map.png'
    plt.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"🖼️ 图 2 已保存: {output2}")
    plt.close(fig2)

if __name__ == "__main__":
    plot_charts()