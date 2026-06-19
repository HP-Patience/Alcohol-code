#!/usr/bin/env python3
"""
审稿修改：综合分析脚本
对应 revision_plan.md 中的 5.1、6.1、6.2
运行方式：在 代码与数据/ 目录下 python scripts/revision_analysis.py
"""
import sys, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '数据文件')
OUT  = os.path.join(BASE, 'revision_results.txt')

np.random.seed(42)

# ============================================================
# 5.1 Elastic Net — 去掉 Session 特征
# ============================================================
print("=" * 60)
print("5.1 Elastic Net 预测模型：去掉 Session")
print("=" * 60)

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score

df_all = pd.read_excel(os.path.join(DATA, 'ML-FlyScore/data_clean.xlsx'))
# 31 个生理/神经特征（不含 session）
X_cols = [c for c in df_all.columns if c not in ('组别','受试者','session','label')]

for grp_name, grp_val in [("Alcohol", 0), ("Control", 1)]:
    df = df_all[df_all['组别'] == grp_val].copy()
    X, y, g = df[X_cols], df['label'], np.array(df['受试者'])

    # GridSearch 找最优 ElasticNet 参数
    pipe = Pipeline([('scaler',StandardScaler()),('model',ElasticNet(max_iter=10000,random_state=42))])
    param = {'model__alpha': np.logspace(-3, 1, 20), 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
    gkf = GroupKFold(8)
    grid = GridSearchCV(pipe, param, cv=gkf, scoring='neg_root_mean_squared_error')
    grid.fit(X, y, groups=g)

    # 手动 CV（避免 sklearn 版本差异）
    y_true_all, y_pred_all = [], []
    for tr, te in gkf.split(X, y, g):
        grid.best_estimator_.fit(X.iloc[tr], y.iloc[tr])
        y_pred_all.append(grid.best_estimator_.predict(X.iloc[te]))
        y_true_all.append(y.iloc[te].values)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # 置换检验（1000 次被试内 shuffle label）
    rng = np.random.default_rng(42)
    cnt = 0
    for p in range(1000):
        yp = y.values.copy()
        for subj in np.unique(g):
            idx = np.where(g == subj)[0]
            yp[idx] = rng.permutation(yp[idx])
        yt, yp2 = [], []
        for tr, te in gkf.split(X, yp, g):
            grid.best_estimator_.fit(X.iloc[tr], yp[tr])
            yt.append(yp[te]); yp2.append(grid.best_estimator_.predict(X.iloc[te]))
        yt = np.concatenate(yt); yp2 = np.concatenate(yp2)
        if r2_score(yt, yp2) >= r2: cnt += 1
    pval = (cnt + 1) / (1000 + 1)

    print(f"\n{grp_name} 组（不含 Session）:")
    print(f"  R² = {r2:.3f}, RMSE = {rmse:.3f}")
    print(f"  置换检验 p = {pval:.4f}（{cnt}/1000 次 >= 真实 R²）")

# ============================================================
# 6.1 缺失分布分析
# ============================================================
print("\n" + "=" * 60)
print("6.1 眼动指标缺失分布：Group × Phase × Session")
print("=" * 60)

df_eye = pd.read_excel(os.path.join(DATA, 'EYE/眼动数据预处理文件.xlsx'))
metrics = ['AOI转换次数', '静态注视熵(SGE)', '眼跳注视熵(GTE)']

# 标记缺失（0 值视为缺失，与 EYE/2 插值脚本一致）
for m in metrics:
    df_eye[f'{m}_miss'] = (df_eye[m] == 0) | (df_eye[m].isna())

for grp in ['A','B']:
    grp_name = 'Alcohol' if grp == 'A' else 'Control'
    print(f"\n  {grp_name}:")
    sub = df_eye[df_eye['组别'] == grp]
    for m in metrics:
        p = sub.groupby(['阶段','天数'])[f'{m}_miss'].mean() * 100
        print(f"    {m}: {p.mean():.1f}%（各 Session: {p.groupby('天数').mean().round(1).to_dict()}）")

# 总体缺失率
print(f"\n  总体缺失率：")
for m in metrics:
    rate = df_eye[f'{m}_miss'].mean() * 100
    print(f"    {m}: {rate:.1f}%")

# ============================================================
# 6.2 无插值敏感性分析（Pairwise Deletion LMM）
# ============================================================
print("\n" + "=" * 60)
print("6.2 无插值敏感性分析：Pairwise Deletion LMM")
print("=" * 60)

import statsmodels.formula.api as smf

df_lmm = pd.read_excel(os.path.join(DATA, 'EYE/眼动数据预处理文件.xlsx'))
df_lmm = df_lmm.rename(columns={'被试者':'受试者','天数':'飞行天数'})
df_lmm['组别'] = df_lmm['组别'].replace({'A':'Alcohol','B':'Control'})
# 合并 4 次转弯
df_lmm['阶段_agg'] = df_lmm['阶段'].replace({
    '起飞阶段':'Takeoff','第1次转弯':'Turning','第2次转弯':'Turning',
    '第3次转弯':'Turning','第4次转弯':'Turning','巡航阶段':'Cruise','降落阶段':'Landing'})
for m in metrics:
    df_lmm[m] = df_lmm[m].replace(0, np.nan)

# 插值结果（来自 EYE/5 原始运行输出）
ip_results = {
    'AOI转换次数': {'beta': -1.892, 'p': 0.044},
    '静态注视熵(SGE)': {'beta': -0.128, 'p': 0.017},
    '眼跳注视熵(GTE)': {'beta': -0.034, 'p': 0.150},
}

for metric_name, col in [
    ('AOI转换次数', 'AOI转换次数'),
    ('静态注视熵(SGE)', '静态注视熵(SGE)'),
    ('眼跳注视熵(GTE)', '眼跳注视熵(GTE)')
]:
    sub = df_lmm.dropna(subset=[col]).copy()
    sub['组别'] = pd.Categorical(sub['组别'], categories=['Alcohol','Control'])
    sub['阶段_agg'] = pd.Categorical(sub['阶段_agg'], categories=['Takeoff','Turning','Cruise','Landing'])

    formula = f"Q('{col}') ~ 组别 * 阶段_agg * 飞行天数"
    model = smf.mixedlm(formula, sub, groups=sub["受试者"])
    result = model.fit(method="powell", maxiter=500, disp=False)

    for param in result.params.index:
        if 'Control' in param and 'Landing' in param and '飞行天数' in param:
            b = result.params[param]
            p = result.pvalues[param]
            ip = ip_results[metric_name]
            match = '✓' if np.sign(ip['beta']) == np.sign(b) else '✗'
            print(f"\n  {metric_name}:")
            print(f"    [插值]   β = {ip['beta']:.3f}, p = {ip['p']:.3f}")
            print(f"    [无插值] β = {b:.3f}, p = {p:.3f}")
            print(f"    方向一致: {match}")

print(f"\n完成。分析脚本已保存至: {os.path.basename(__file__)}")
