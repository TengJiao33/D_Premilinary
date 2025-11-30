import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import os

def safe_wkt_load(wkt_string):
    """
    尝试解析 WKT 字符串。
    如果成功，返回几何对象；
    如果失败（比如没闭合），返回 None。
    """
    try:
        return wkt.loads(wkt_string)
    except Exception:
        return None

def draw_manhattan_map_robust():
    csv_file_path = './raw_data/DSNY_Districts_20251130.csv' # 请确保路径正确
    
    print(f"📂 正在读取: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print("❌ 文件不存在")
        return

    # 1. 读取数据
    df = pd.read_csv(csv_file_path, usecols=['DISTRICT', 'multipolygon'])
    
    # 2. 筛选曼哈顿 (带空值保护)
    df_mn = df[df['DISTRICT'].str.startswith('MN', na=False)].copy()
    print(f"🔍 找到 {len(df_mn)} 个曼哈顿分区行。")

    # 3. 【关键修改】容错解析
    print("⚙️ 正在解析几何数据 (自动跳过损坏行)...")
    
    # 对每一行尝试解析，坏的变成 None
    df_mn['geometry'] = df_mn['multipolygon'].apply(safe_wkt_load)
    
    # 分离出成功和失败的
    valid_districts = df_mn[df_mn['geometry'].notna()]
    failed_districts = df_mn[df_mn['geometry'].isna()]
    
    print(f"✅ 成功解析: {len(valid_districts)} 个")
    print(f"❌ 解析失败: {len(failed_districts)} 个")
    
    if len(failed_districts) > 0:
        print("⚠️ 以下分区的地图数据已损坏 (将被跳过):")
        print(failed_districts['DISTRICT'].tolist())

    if len(valid_districts) == 0:
        print("🔴 所有分区数据都损坏了，无法绘图。")
        return

    # 4. 绘图 (只画能画的)
    gdf = gpd.GeoDataFrame(valid_districts, geometry='geometry')
    
    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(ax=ax, color='#ADD8E6', edgecolor='black', alpha=0.8)

    # 标注名字
    for idx, row in gdf.iterrows():
        try:
            centroid = row['geometry'].centroid
            ax.annotate(text=row['DISTRICT'], 
                        xy=(centroid.x, centroid.y), 
                        ha='center', fontsize=9, fontweight='bold', color='darkred')
        except:
            pass # 如果算不出中心点就不标了

    plt.title(f"Manhattan Districts ({len(valid_districts)}/{len(df_mn)} Visible)", fontsize=15)
    plt.axis('off')
    
    output_file = 'manhattan_map_robust.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"🖼️ 图片已保存: {output_file}")
    plt.show()

if __name__ == "__main__":
    draw_manhattan_map_robust()