import pandas as pd


def calculate_real_ratios(csv_path):
    print(f"正在读取 {csv_path} ...")
    # 读取 PLUTO 数据 (可能很大，只读关键列)
    # 关键列推测: 'CD' (Community District), 'UnitsRes' (Residential Units)
    try:
        df = pd.read_csv(csv_path, usecols=['CD', 'UnitsRes'])
    except ValueError:
        # 如果列名不对，尝试读取前几行看看
        df = pd.read_csv(csv_path, nrows=5)
        print("列名列表:", df.columns.tolist())
        return None

    print("数据加载成功，正在计算小楼比例...")

    # 过滤曼哈顿的数据 (CD 以 1 开头，如 101, 102)
    # 你的 MN.csv 可能已经是曼哈顿的了，但还是保险起见
    df['CD'] = pd.to_numeric(df['CD'], errors='coerce')
    manhattan_df = df[(df['CD'] >= 101) & (df['CD'] <= 112)].copy()

    # 定义 "Bin-Compatible" (可以用桶的小楼): 1 <= UnitsRes <= 9
    manhattan_df['Is_Small_Building'] = (manhattan_df['UnitsRes'] >= 1) & (manhattan_df['UnitsRes'] <= 9)

    # 按 CD 分组统计
    # Count: 总建筑数
    # Sum: 小楼数量
    stats = manhattan_df.groupby('CD')['Is_Small_Building'].agg(['sum', 'count'])
    stats['Ratio'] = stats['sum'] / stats['count']

    print("\n=== 基于 MN.csv 算出的真实普及率 (Adoption Rate) ===")
    print(stats)

    # 生成字典代码供 copy
    print("\n请把下面这个字典复制到 solve5.py 中替换 REAL_BIN_ADOPTION_STOPS:")
    print("REAL_BIN_ADOPTION_STOPS = {")
    for cd, row in stats.iterrows():
        print(f"    {int(cd)}: {row['Ratio']:.3f},")
    print("}")


# 运行
calculate_real_ratios('extra_data/building_data/MN.csv')