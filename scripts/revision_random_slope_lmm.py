#!/usr/bin/env python3
"""5.2 随机斜率混合模型：允许个体学习速率不同"""
import sys, os, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE = "f:/claudecode/五边飞行paper/五边飞行paper/代码与数据"
DATA = f"{BASE}/数据文件"

print("=" * 60)
print("5.2 随机斜率混合模型分析")
print("=" * 60)

# Read flight score data
df = pd.read_excel(f"{DATA}/ML-FlyScore/data_clean.xlsx")
df['组别'] = df['组别'].replace({0:'Alcohol',1:'Control'})
df['组别'] = pd.Categorical(df['组别'], ['Alcohol','Control'])
print(f"Data: {len(df)} rows, {df['受试者'].nunique()} subjects")

# Model 1: Random intercept only (original)
print("\n--- Model 1: Random Intercept Only ---")
formula_ri = "label ~ 组别 * session"
model_ri = smf.mixedlm(formula_ri, df, groups=df["受试者"],
                       re_formula="1")  # random intercept only
result_ri = model_ri.fit(method="powell", maxiter=500, disp=False)
print(result_ri.summary())

# Extract key stats
ri_loglik = result_ri.llf
ri_aic = result_ri.aic
ri_bic = result_ri.bic
ri_params = result_ri.params
ri_pvals = result_ri.pvalues
ri_group_var = result_ri.cov_re.iloc[0,0]  # random intercept variance
ri_resid_var = result_ri.scale  # residual variance
ri_icc = ri_group_var / (ri_group_var + ri_resid_var)
print(f"\n  LogLik={ri_loglik:.1f}, AIC={ri_aic:.0f}, BIC={ri_bic:.0f}")
print(f"  Group Var (intercept)={ri_group_var:.3f}, Residual={ri_resid_var:.3f}, ICC={ri_icc:.3f}")
print(f"  Group (组别): β={ri_params['组别[T.Control]']:.3f}, p={ri_pvals['组别[T.Control]']:.4f}")
print(f"  Group×Session: β={ri_params['组别[T.Control]:session']:.3f}, p={ri_pvals['组别[T.Control]:session']:.4f}")

# Model 2: Random intercept + random slope for session
print("\n\n--- Model 2: Random Intercept + Random Slope ---")
model_rs = smf.mixedlm(formula_ri, df, groups=df["受试者"],
                       re_formula="session")  # random intercept + slope
result_rs = model_rs.fit(method="powell", maxiter=500, disp=False)
print(result_rs.summary())

rs_loglik = result_rs.llf
rs_aic = result_rs.aic
rs_bic = result_rs.bic
rs_params = result_rs.params
rs_pvals = result_rs.pvalues
rs_re = result_rs.cov_re
print(f"\n  LogLik={rs_loglik:.1f}, AIC={rs_aic:.0f}, BIC={rs_bic:.0f}")
print(f"  Random effects covariance matrix:")
print(f"    {rs_re}")
print(f"  Key fixed effects:")
print(f"    Group (组别): β={rs_params['组别[T.Control]']:.3f}, p={rs_pvals['组别[T.Control]']:.4f}")
print(f"    Group×Session: β={rs_params['组别[T.Control]:session']:.3f}, p={rs_pvals['组别[T.Control]:session']:.4f}")

# Likelihood Ratio Test
print("\n\n--- Likelihood Ratio Test ---")
lr_stat = -2 * (ri_loglik - rs_loglik)
lr_df = 2  # random slope adds 2 parameters (variance + covariance)
lr_p = 1 - sm.distributions.chi2.cdf(lr_stat, lr_df)
print(f"  LRT χ²({lr_df}) = {lr_stat:.2f}, p = {lr_p:.4f}")

# Store results
with open(f"{DATA}/随机斜率模型结果.txt", 'w', encoding='utf-8') as f:
    f.write(f"=== 随机斜率混合模型结果 ===\n\n")
    f.write(f"Model 1 (Random Intercept):\n")
    f.write(f"  LogLik={ri_loglik:.1f}, AIC={ri_aic:.0f}, BIC={ri_bic:.0f}\n")
    f.write(f"  Group Var={ri_group_var:.3f}, Residual={ri_resid_var:.3f}, ICC={ri_icc:.3f}\n")
    f.write(f"  Group: β={ri_params['组别[T.Control]']:.3f}, p={ri_pvals['组别[T.Control]']:.4f}\n")
    f.write(f"  Group×Session: β={ri_params['组别[T.Control]:session']:.3f}, p={ri_pvals['组别[T.Control]:session']:.4f}\n\n")
    f.write(f"Model 2 (Random Intercept + Random Slope):\n")
    f.write(f"  LogLik={rs_loglik:.1f}, AIC={rs_aic:.0f}, BIC={rs_bic:.0f}\n")
    f.write(f"  Random effects:\n")
    f.write(f"    {rs_re}\n")
    f.write(f"  Group: β={rs_params['组别[T.Control]']:.3f}, p={rs_pvals['组别[T.Control]']:.4f}\n")
    f.write(f"  Group×Session: β={rs_params['组别[T.Control]:session']:.3f}, p={rs_pvals['组别[T.Control]:session']:.4f}\n\n")
    f.write(f"Likelihood Ratio Test:\n")
    f.write(f"  χ²({lr_df}) = {lr_stat:.2f}, p = {lr_p:.4f}\n")

print(f"\nResults saved to {DATA}/随机斜率模型结果.txt")
print("Done!")
