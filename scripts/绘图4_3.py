import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import platform
import numpy as np

# ================= 1. 基础配置 =================
MAP_FILE = './raw_data/DSNY_Districts_20251130.csv'
DATA_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'

def configure_style():
    plt.style.use('default') 
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 数据处理 =================
def load_and_fix_data():
    # A. 加载地图
    if not os.path.exists(MAP_FILE): 
        print(f"❌ 地图文件未找到: {MAP_FILE}")
        return None
    
    df_map = pd.read_csv(MAP_FILE)
    df_map = df_map[df_map['DISTRICT'].str.startswith('MN', na=False)].copy()
    
    # 容错解析 WKT
    def safe_wkt(x):
        try: return wkt.loads(x)
        except: return None
        
    df_map['geometry'] = df_map['multipolygon'].apply(safe_wkt)
    gdf = gpd.GeoDataFrame(df_map.dropna(subset=['geometry']), geometry='geometry')
    
    # B. 加载数据
    if not os.path.exists(DATA_FILE): 
        print(f"❌ 数据文件未找到: {DATA_FILE}")
        return None
        
    df_data = pd.read_csv(DATA_FILE)
    df_data = df_data.dropna(subset=['Rat_Complaints', 'Monthly_Trash_Tons'])
    
    # C. 计算策略 (AM/PM)
    rat_threshold = df_data['Rat_Complaints'].quantile(0.60)
    
    # 使用布尔逻辑避免字符串错误
    df_data['Is_AM'] = df_data['Rat_Complaints'] >= rat_threshold
    df_data['Strategy_Label'] = df_data['Is_AM'].apply(
        lambda x: 'AM Strategy (Morning)' if x else 'PM Strategy (Evening)'
    )
    
    # D. 计算风险影响 (Impact Stats)
    daily_tons = df_data['Monthly_Trash_Tons'] / 30.0
    baseline_risk = (daily_tons * 22.0).sum()
    
    # AM暴露11h, PM暴露22h
    risk_am = daily_tons[df_data['Is_AM']].sum() * 11.0
    risk_pm = daily_tons[~df_data['Is_AM']].sum() * 22.0
    optimized_risk = risk_am + risk_pm
    
    impact_stats = (baseline_risk, optimized_risk)

    # E. 合并
    # 统一 ID 格式 MN01
    def clean_id(x):
        try: return f"MN{int(x)%100:02d}"
        except: return str(x)
        
    df_data['DISTRICT'] = df_data['CD_ID'].apply(clean_id)
    merged = gdf.merge(df_data, on='DISTRICT', how='left')
    
    # 计算中心点
    merged['centroid'] = merged.geometry.centroid
    
    return merged, impact_stats

# ================= 3. 绘图主程序 =================
def draw_dashboard_fixed_v2():
    configure_style()
    data = load_and_fix_data()
    if data is None: return
    
    gdf, (base_risk, opt_risk) = data
    
    # 创建画布
    fig = plt.figure(figsize=(12, 10), dpi=300) # 尺寸稍微改小一点点防止内存压力
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('white')
    
    # --- Layer 1: 战略底图 ---
    color_am = '#F4D03F' # 亮金
    color_pm = '#2E4053' # 深岩灰蓝
    
    # 绘制 PM
    pm_zone = gdf[gdf['Strategy_Label'] == 'PM Strategy (Evening)']
    if not pm_zone.empty:
        pm_zone.plot(ax=ax, color=color_pm, edgecolor='white', linewidth=0.8, alpha=0.95)
        
    # 绘制 AM
    am_zone = gdf[gdf['Strategy_Label'] == 'AM Strategy (Morning)']
    if not am_zone.empty:
        am_zone.plot(ax=ax, color=color_am, edgecolor='white', linewidth=0.8, alpha=0.95)
    
    # --- Layer 2: 气泡 ---
    max_rats = gdf['Rat_Complaints'].max()
    # 气泡大小归一化 (根据经纬度尺度调整，0.01度约等于1km)
    # 这里我们用 scatter 的 s 参数，它是 points^2
    # 需要把数值映射到合适的 s 大小，比如 50~500
    gdf['bubble_size'] = gdf['Rat_Complaints'] / max_rats * 1000
    
    x = gdf['centroid'].x
    y = gdf['centroid'].y
    
    # 白底光晕
    ax.scatter(x, y, s=gdf['bubble_size'] + 60, c='white', alpha=0.8, zorder=9)
    # 红气泡
    ax.scatter(x, y, s=gdf['bubble_size'], 
               c='#E74C3C', alpha=0.76, 
               edgecolor='#C0392B', linewidth=1.0, zorder=10)
    

    # --- Layer 4: 悬浮柱状图 ---
    # 调整位置 [left, bottom, width, height]
    rect = [0.15, 0.60, 0.25, 0.2] 
    ax_inset = fig.add_axes(rect)
    
    colors = ['#95a5a6', '#27ae60']
    bars = ax_inset.bar(['Baseline', 'Optimized'], [base_risk, opt_risk], 
                        color=colors, width=0.5)
    
    # 计算并展示真实的减少率
    if base_risk > 0:
        reduction = (base_risk - opt_risk) / base_risk * 100
    else:
        reduction = 0
        
    ax_inset.set_title(f"Impact Assessment\nTrash-Exposure Reduced by {reduction:.1f}%", 
                       fontsize=10, fontweight='bold', color='#2c3e50')
    
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['left'].set_visible(False)
    ax_inset.set_yticks([])
    
    for bar in bars:
        height = bar.get_height()
        ax_inset.text(bar.get_x() + bar.get_width()/2, height, 
                      f"{int(height):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- Layer 5: 修饰 ---
    ax.axis('off')
    ax.set_title("Strategic Command Dashboard: Rodent Mitigation Plan", 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 图例
    patch_am = mpatches.Patch(color=color_am, label='AM Shift Zone (High Risk)')
    patch_pm = mpatches.Patch(color=color_pm, label='PM Shift Zone (Low Risk)')
    line_bubble = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='#E74C3C', 
                                markersize=10, markeredgecolor='#C0392B', label='Rat Intensity')
    
    ax.legend(handles=[patch_am, patch_pm, line_bubble], 
              loc='lower right', frameon=True, fontsize=10, 
              title="Legend", title_fontsize=11,
              bbox_to_anchor=(0.95, 0.05))

    output_file = 'Final_Dashboard_Fixed_v2.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 成功生成: {output_file}")
    plt.show()

if __name__ == "__main__":
    draw_dashboard_fixed_v2()