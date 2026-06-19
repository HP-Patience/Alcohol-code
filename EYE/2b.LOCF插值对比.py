#!/usr/bin/env python3
"""LOCF插值 + 三列对比分析（线性插值 vs LOCF vs 无插值）"""
import sys, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf

BASE = "f:/claudecode/五边飞行paper/五边飞行paper/代码与数据"
DATA = f"{BASE}/数据文件"

# ============================================================
# 1. Read pre-interpolation data
# ============================================================
print("1. Reading data...")
df = pd.read_excel(f"{DATA}/EYE/眼动数据预处理文件.xlsx")
df = df.rename(columns={'被试者':'受试者','天数':'飞行天数'})
df['组别'] = df['组别'].replace({'A':'Alcohol','B':'Control'})
metrics = ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']

# Replace 0 with NaN
for m in metrics:
    df[m] = df[m].replace(0, np.nan)

print(f"   Original: {len(df)} rows, missing: {df[metrics].isna().sum().sum()}")

# ============================================================
# 2. LOCF interpolation
# ============================================================
print("\n2. Applying LOCF interpolation...")
# Sort by (受试者, 阶段, 飞行天数) then forward-fill within each group
df_locf = df.copy()
df_locf = df_locf.sort_values(['受试者', '阶段', '飞行天数'])
for m in metrics:
    df_locf[m] = df_locf.groupby(['受试者', '阶段'])[m].transform(lambda x: x.ffill().bfill())

locf_missing = df_locf[metrics].isna().sum().sum()
print(f"   After LOCF: {locf_missing} remaining NaN (no prior value to carry)")

# ============================================================
# 3. Run LMMs for LOCF data
# ============================================================
print("\n3. Running LMMs with LOCF data...")
phase_map = {'起飞阶段':'Takeoff','第1次转弯':'Turning','第2次转弯':'Turning',
             '第3次转弯':'Turning','第4次转弯':'Turning','巡航阶段':'Cruise','降落阶段':'Landing'}
df_locf['阶段_agg'] = df_locf['阶段'].replace(phase_map)
df_locf['阶段_agg'] = pd.Categorical(df_locf['阶段_agg'], ['Takeoff','Turning','Cruise','Landing'])
df_locf['组别_c'] = pd.Categorical(df_locf['组别'], ['Alcohol','Control'])

results_locf = {}
for metric_name, col in [('AOI','AOI转换次数'),('SGE','静态注视熵(SGE)'),('GTE','眼跳注视熵(GTE)')]:
    formula = f"Q('{col}') ~ 组别_c * 阶段_agg * 飞行天数"
    model = smf.mixedlm(formula, df_locf, groups=df_locf["受试者"])
    result = model.fit(method="powell", maxiter=500, disp=False)
    for param in result.params.index:
        if 'Control' in param and 'Landing' in param and '飞行天数' in param:
            results_locf[metric_name] = {'beta': result.params[param], 'p': result.pvalues[param]}
            print(f"   {metric_name}: beta={result.params[param]:.3f}, p={result.pvalues[param]:.4f}")

# ============================================================
# 4. Three-way comparison table
# ============================================================
print("\n4. Three-way comparison table:")

# Linear interpolation results (from notebook EYE/5)
ip = {'AOI':(-1.892, 0.044), 'SGE':(-0.128, 0.017), 'GTE':(-0.034, 0.150)}

# No-imputation results (from revision_analysis.py)
ni = {'AOI':(-1.527, 0.150), 'SGE':(-0.022, 0.694), 'GTE':(-0.005, 0.854)}

print(f"\n{'Metric':<10} {'Interp β':<12} {'Interp p':<12} {'LOCF β':<12} {'LOCF p':<12} {'No-imp β':<12} {'No-imp p':<12} {'Match':<8}")
print("-" * 90)
for metric in ['AOI','SGE','GTE']:
    ip_b, ip_p = ip[metric]
    ni_b, ni_p = ni[metric]
    l_b, l_p = results_locf[metric]['beta'], results_locf[metric]['p']
    match = '✓' if np.sign(ip_b) == np.sign(l_b) == np.sign(ni_b) else '✗'
    print(f"{metric:<10} {ip_b:<12.3f} {ip_p:<12.3f} {l_b:<12.3f} {l_p:<12.3f} {ni_b:<12.3f} {ni_p:<12.3f} {match:<8}")

# ============================================================
# 5. Save results
# ============================================================
df_locf.to_excel(f"{DATA}/EYE/眼动数据LOCF插值文件.xlsx", index=False)
comparison = []
for metric in ['AOI','SGE','GTE']:
    ip_b, ip_p = ip[metric]
    ni_b, ni_p = ni[metric]
    l_b, l_p = results_locf[metric]['beta'], results_locf[metric]['p']
    comparison.append({
        '指标': metric, '效应': 'Control:Landing:Session',
        '线性插值_β': ip_b, '线性插值_p': ip_p,
        'LOCF_β': l_b, 'LOCF_p': l_p,
        '无插值_β': ni_b, '无插值_p': ni_p,
        '方向一致': '✓' if np.sign(ip_b)==np.sign(l_b)==np.sign(ni_b) else '✗'
    })
pd.DataFrame(comparison).to_excel(f"{DATA}/EYE/LOCF_vs_插值对比.xlsx", index=False)
print(f"\nDone! Results saved.")
