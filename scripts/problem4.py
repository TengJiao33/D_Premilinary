import pandas as pd
import numpy as np

# ================= 配置与 L5 数据加载 =================
INPUT_FILE = 'extra_data/merged_data/Manhattan_Data_Current_2023_2025.csv'

def load_data_l5(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Rat_Complaints', 'Monthly_Trash_Tons', 'CD_ID'])
    return df

def analyze_problem4_rats(df):
    
    # --- 1. 调查相关性---
    # 计算皮尔逊相关系数
    corr = df['Rat_Complaints'].corr(df['Monthly_Trash_Tons'])
    
    # --- 2. 制定早晚班策略 (Designate AM/PM) [Cite: 27] ---
    # 策略：鼠患越严重，越必须在早上收 (减少暴露时间)
    # 计算阈值：前 40% 严重的区域指定为 AM (这通常覆盖了所有的 High Risk)
    rat_threshold = df['Rat_Complaints'].quantile(0.6) # Top 40%
    
    def assign_shift(row):
        if row['Rat_Complaints'] >= rat_threshold:
            return 'AM (Morning)'  # 高鼠患 -> 必须早收
        else:
            return 'PM (Evening)'  # 低鼠患 -> 可以晚收
            
    df['Assigned_Shift'] = df.apply(assign_shift, axis=1)
    
    # --- 3. 评估对老鼠种群的影响 (Effect on Rat Population) [Cite: 26] ---
    # 建立模型指标: Trash-Exposure-Hours (TEH)
    # TEH = Daily_Tons * Exposure_Hours
    # 假设 Baseline (现状): 混乱或主要是 PM，平均暴露 20小时
    # 假设 Optimized (新策略): AM 暴露 11小时 (8pm-7am), PM 暴露 22小时 (8pm-6pm)
    
    # 估算日均垃圾量
    df['Daily_Tons'] = df['Monthly_Trash_Tons'] / 30.0
    
    # 计算 Baseline 风险 (假设全城平均暴露 20h)
    baseline_teh = (df['Daily_Tons'] * 20.0).sum()
    
    # 计算 Optimized 风险
    def calc_new_exposure(row):
        if row['Assigned_Shift'] == 'AM (Morning)':
            return row['Daily_Tons'] * 11.0 # 暴露时间减半
        else:
            return row['Daily_Tons'] * 22.0 # 暴露时间正常
            
    optimized_teh = df.apply(calc_new_exposure, axis=1).sum()
    
    # 计算改善率
    reduction = baseline_teh - optimized_teh
    reduction_pct = (reduction / baseline_teh) * 100
    
    return corr, df, baseline_teh, optimized_teh, reduction_pct

# ================= 结果输出 =================
if __name__ == "__main__":
    df = load_data_l5(INPUT_FILE)
    corr, df_res, base_risk, opt_risk, imp_pct = analyze_problem4_rats(df)
    
    
    # Q1: 相关性
    print(f"\n[Q1] 垃圾与老鼠的相关性")
    print(f"   Pearson r = {corr:.4f}")

    # Q3: 早晚班名单
    print(f"\n[Q3] 区域时间分配")
    am_zones = df_res[df_res['Assigned_Shift'] == 'AM (Morning)']
    pm_zones = df_res[df_res['Assigned_Shift'] == 'PM (Evening)']
    
    print(f"   已指定 {len(am_zones)} 个区域进行早晨收集:")
    print(f"   目标: 覆盖了全城 {am_zones['Rat_Complaints'].sum() / df_res['Rat_Complaints'].sum():.1%} 的鼠患投诉。")
    print(f"   AM 区域示例: {am_zones['CD_ID'].head(5).tolist()} ")
    
    print(f"   已指定 {len(pm_zones)} 个区域进行傍晚收集 :")

    # Q2: 对老鼠种群的影响
    print(f"\n[Q2] 对老鼠种群的预期影响")
    print(f"   - 现状: {int(base_risk):,} TEH")
    print(f"   - 策略后: {int(opt_risk):,} TEH")
    print(f"   老鼠的食物源暴露机会降低 {imp_pct:.1f}%。")
    print("="*50)