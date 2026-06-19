#!/usr/bin/env python3
"""FDR correction: collect all LMM p-values across modalities, apply BH correction"""
import sys, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE = "f:/claudecode/五边飞行paper/五边飞行paper/代码与数据"
DATA = f"{BASE}/数据文件"

results = []  # (modality, metric, effect, p_value)

# ============================================================
# 1. EYE - read from existing LMM result file
# ============================================================
print("1. EYE LMMs...")
eye_file = f"{DATA}/EYE/混合效应模型结果.xlsx"
for metric in ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']:
    df = pd.read_excel(eye_file, sheet_name=metric)
    for _, row in df.iterrows():
        term = str(row['Unnamed: 0'])
        pval = row['p_value']
        # Only keep interaction terms (those with ':')
        if ':' in term:
            results.append({
                'modality': 'EYE',
                'metric': metric,
                'effect': term,
                'p_value': pval
            })
print(f"   Collected {len([r for r in results if r['modality']=='EYE'])} p-values")

# ============================================================
# 2. GSR - aggregate sample-level data, run LMM
# ============================================================
print("2. GSR LMM...")
gsr = pd.read_excel(f"{DATA}/GSR/GSR_SCL_AllEvents_pre1sBaseline_3sd.xlsx")

# Fix: GSR phases are coded as 'qifei','1','2','3','4','jiangluo','hejiu'
# There's no 'Cruise' phase in GSR data - all turns are numeric codes
gsr_phase_map = {'qifei':'Takeoff', '1':'Turning', '2':'Turning', '3':'Turning',
                 '4':'Turning', 'jiangluo':'Landing', 'hejiu':'Resting'}
gsr['阶段_en'] = gsr['阶段'].map(gsr_phase_map)
gsr['组别'] = gsr['组别'].replace({'A':'Alcohol','B':'Control'})

# Aggregate to (subject, session, phase) means
gsr_agg = gsr.groupby(['组别','姓名','飞行天数','阶段_en'], as_index=False)['data'].mean()
gsr_agg = gsr_agg.rename(columns={'姓名':'受试者','data':'SCL','阶段_en':'阶段'})
gsr_agg['组别'] = pd.Categorical(gsr_agg['组别'], ['Alcohol','Control'])
gsr_agg['阶段'] = pd.Categorical(gsr_agg['阶段'], ['Takeoff','Turning','Landing','Resting'])

print(f"   Aggregated: {len(gsr_agg)} rows, {gsr_agg['受试者'].nunique()} subjects")

if len(gsr_agg) > 0:
    try:
        # Use Resting as reference (alphabetically first if we specify)
        formula = "SCL ~ 组别 * 阶段 * 飞行天数"
        model = smf.mixedlm(formula, gsr_agg, groups=gsr_agg["受试者"])
        result = model.fit(method="powell", maxiter=500, disp=False)
        for param in result.params.index:
            pval = result.pvalues[param]
            results.append({
                'modality': 'GSR', 'metric': 'SCL',
                'effect': param, 'p_value': pval
            })
        print(f"   Collected {len([r for r in results if r['modality']=='GSR'])} p-values")
    except Exception as e:
        print(f"   GSR FAILED: {e}")

# ============================================================
# 3. PPG session-level LMM
# ============================================================
print("3. PPG LMMs (session-level)...")
ml = pd.read_excel(f"{DATA}/ML-FlyScore/data_clean.xlsx")
ml['组别'] = ml['组别'].replace({0:'Alcohol',1:'Control'})
ml['组别'] = pd.Categorical(ml['组别'], ['Alcohol','Control'])

for metric in ['HR','SDNN','RMSSD']:
    try:
        formula = f"{metric} ~ 组别 * session"
        model = smf.mixedlm(formula, ml, groups=ml["受试者"])
        result = model.fit(method="powell", maxiter=500, disp=False)
        for param in result.params.index:
            if param != 'Intercept':
                results.append({
                    'modality': 'PPG', 'metric': metric,
                    'effect': param, 'p_value': result.pvalues[param]
                })
    except Exception as e:
        print(f"   PPG {metric}: {e}")
print(f"   Collected {len([r for r in results if r['modality']=='PPG'])} p-values")

# ============================================================
# 4. EEG - use existing beta power and DE data
# ============================================================
print("4. EEG LMMs...")

# EEG power (beta): data_clean has beta_high and beta_low
for metric in [c for c in ml.columns if 'β' in c or 'DE' in c]:
    try:
        formula = f"Q('{metric}') ~ 组别 * session"
        model = smf.mixedlm(formula, ml, groups=ml["受试者"])
        result = model.fit(method="powell", maxiter=500, disp=False)
        for param in result.params.index:
            if ':' in param or (param not in ['Intercept','Group'] and '组别' in param):
                results.append({
                    'modality': 'EEG', 'metric': metric,
                    'effect': param, 'p_value': result.pvalues[param]
                })
    except Exception as e:
        pass  # skip failed metrics silently
print(f"   Collected {len([r for r in results if r['modality']=='EEG'])} p-values")

# ============================================================
# 5. Apply BH correction per modality
# ============================================================
print("\n" + "=" * 80)
print("FDR Correction (Benjamini-Hochberg) per Modality")
print("=" * 80)

df_results = pd.DataFrame(results)
all_output = []

for modality in ['EYE','GSR','PPG','EEG']:
    sub = df_results[df_results['modality']==modality].copy()
    if len(sub) == 0:
        print(f"\n{modality}: No data")
        continue

    pvals = sub['p_value'].values
    rejected, p_corrected, _, _ = multipletests(pvals, method='fdr_bh')

    sub['p_corrected'] = p_corrected
    sub['survives_fdr'] = rejected
    sub['sig_mark'] = sub['survives_fdr'].apply(lambda x: '***' if x else '')

    n_sig = rejected.sum()
    print(f"\n{modality}: {len(sub)} tests, {n_sig} survive FDR correction")

    for _, row in sub.iterrows():
        mark = '***' if row['survives_fdr'] else ''
        if row['p_value'] < 0.1 or row['survives_fdr']:
            print(f"  {row['effect'][:45]:45s}  p={row['p_value']:.4f}  adj-p={row['p_corrected']:.4f}  {mark}")

    all_output.append(sub)

# ============================================================
# 6. Save results
# ============================================================
df_full = pd.concat(all_output, ignore_index=True)
out_file = f"{DATA}/FDR校正结果.xlsx"
df_full.to_excel(out_file, index=False)
print(f"\nFull results saved to: {out_file}")
print(f"\nTotal p-values collected: {len(df_full)}")
print(f"Total surviving FDR: {df_full['survives_fdr'].sum()}")
