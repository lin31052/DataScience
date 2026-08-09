"""
电商零售预测 Step 3 — 序列诊断 (数理统计): 该序列到底可预测性多高?
目标: 用 ACF/PACF/季节分解/信噪比 判断周度销售序列的可预测结构和瓶颈,
      决定后续建模该投哪些方向 (改粒度/加季节/换模型)。
数据: UCI Online Retail 清洗后 UK 53 周销售额 + 订单量
"""
import pandas as pd
import numpy as np
import os, glob, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

def setup_chinese_font():
    ff = []
    for pat in ["/usr/share/fonts/**/wqy-microhei.ttc", "/usr/share/fonts/**/wqy-zenhei.ttc",
                "/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/*CJK*.ttc",
                "/usr/share/fonts/**/*WenQuanYi*"]:
        ff += glob.glob(pat, recursive=True)
    for fp in sorted(set(ff)):
        try: fm.fontManager.addfont(fp)
        except Exception: pass
    zh = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "Noto Sans CJK SC"]
    avail = [n for n in zh if any(f.name == n for f in fm.fontManager.ttflist)]
    if not avail:
        avail = sorted({f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("CJK","WenQuanYi"))})
    return avail

zh_fonts = setup_chinese_font()
if zh_fonts:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = zh_fonts + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.getcwd() + "/test/retail_sales/output"
os.makedirs(OUT, exist_ok=True)
CSV = os.getcwd() + "/test/retail_sales/online_retail.csv"

# ---------- 读+清洗+周度聚合 ----------
df = pd.read_csv(CSV, parse_dates=["InvoiceDate"])
df_uk = df[df["Country"] == "United Kingdom"].copy()
df_uk = df_uk[df_uk["CustomerID"].notna()]
df_uk = df_uk[df_uk["Quantity"] > 0]
df_uk = df_uk[~df_uk["InvoiceNo"].astype(str).str.startswith("C")]
df_uk["Revenue"] = df_uk["Quantity"] * df_uk["UnitPrice"]
df_uk["Week"] = df_uk["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time)
wk = df_uk.groupby("Week").agg(Revenue=("Revenue","sum"), Orders=("InvoiceNo","nunique")).sort_index()

rev = wk["Revenue"]
print("=" * 64)
print("电商周度销售序列 诊断报告")
print("=" * 64)
print(f"序列长度: {len(rev)} 周 ({rev.index.min().date()} ~ {rev.index.max().date()})")

# ---------- 1. 描述统计 + 正态性 ----------
from scipy import stats
print(f"\n【1. 基础统计】")
print(f" 均值={rev.mean():,.0f}  标准差={rev.std():,.0f}  变异系数CV={rev.std()/rev.mean():.3f}")
print(f"  最小值={rev.min():,.0f}  中位数={rev.median():,.0f}  最大值={rev.max():,.0f}")
sw_stat, sw_p = stats.shapiro(rev)
print(f"  Shapiro正态性检验: W={sw_stat:.3f} p={sw_p:.4f} ({'非正态' if sw_p<0.05 else '可认为正态'})")
print(f"  偏度={rev.skew():.3f} 峰度={rev.kurt():.3f}  (右偏+尖峰=有大额周/旺季)")

# ---------- 2. 平稳性: ADF 检验 ----------
# 用 statsmodels
from statsmodels.tsa.stattools import adfuller
adf, adf_p, *_ = adfuller(rev, autolag="AIC")
print(f"\n【2. 平稳性 ADF检验】")
print(f"  ADF_stat={adf:.3f}  p值={adf_p:.4f}  → {'序列平稳(可作回归)' if adf_p<0.05 else '非平稳! 需差分/去趋势'}")
# 一阶差分
adf_d, adf_dp, *_ = adfuller(rev.diff().dropna(), autolag="AIC")
print(f"  一阶差分后 ADF p={adf_dp:.4f} → {'已平稳' if adf_dp<0.05 else '仍非平稳'}")

# ---------- 3. 自相关结构: ACF / PACF ----------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
nlag = min(20, len(rev)//2)
acf_vals = []
for k in range(1, nlag+1):
    # 手工皮尔逊 lag-k 自相关: corr(rev_t, rev_{t-k})
    a = rev[k:].values
    b = rev[:-k].values
    r = np.corrcoef(a, b)[0,1]
    acf_vals.append(r)
print(f"\n【3. 自相关结构 ACF(滞后1-{nlag})】")
for k in range(0, 8):
    if k < len(acf_vals):
        sig = "***" if abs(acf_vals[k]) > 1.96/np.sqrt(len(rev)) else ""
        print(f"  lag{k+1}: r={acf_vals[k]:+.3f}{sig}")
print(f"  (95%显著边界 ≈ ±{1.96/np.sqrt(len(rev)):.3f})")
print("  结论: 若前几个lag显著 → 滞后特征有效; 若全不显著 → 纯滞后难建模(需外部信息)")

# ---------- 4. 季节分解 (STL) ----------
from statsmodels.tsa.seasonal import seasonal_decompose
try:
    decomp = seasonal_decompose(rev, model="additive", period=12)  # 周期设12周试探季度性
    trend = decomp.trend.dropna()
    seas = decomp.seasonal.dropna()
    resid = decomp.resid.dropna()
    vari_obs = rev.var()
    print(f"\n【4. 季节分解 (additive, 周期=12周)】")
    print(f"  趋势分量方差占比={trend.var()/vari_obs:.2f}  季节={seas.var()/vari_obs:.2f}  残差={resid.var()/vari_obs:.2f}")
    print(f"  (残差方差占比越低 → 结构越强越可预测; 残差>0.5 = 噪声太大难建模)")
    base_decomp = f"  趋势{trend.var()/vari_obs:.2f}/季节{seas.var()/vari_obs:.2f}/残差{resid.var()/vari_obs:.2f}"
except Exception as e:
    base_decomp = f"  (seasonal_decompose失败: {e})"
    print(base_decomp)

# ---------- 5. 季节强度 (从ACF里lag-12/lag-4) ----------
print(f"\n【5. 可能的季节/周期】")
for p in (4, 12, 24, 52):
    if p < len(acf_vals):
        print(f"  lag{p} 自相关 = {acf_vals[p-1]:+.3f}  ({'有周期信号' if abs(acf_vals[p-1])>0.4 else '弱'})")

# ---------- 6. 信噪比 / 可预测性下界 ----------
# 朴素预测: 用上一周值预测本周 (naive)
naive_mae = np.mean(np.abs(np.diff(rev)))
naive_rmse = np.sqrt(np.mean(np.diff(rev)**2))
print(f"\n【6. 朴素基准 (persistence: 用上周值预测本周)】")
print(f"  Naive MAE={naive_mae:,.0f}  Naive RMSE={naive_rmse:,.0f}")
print(f"  (一个好的模型至少得打赢这个朴素基准; 若打不赢说明滞后方法上限极低)")

# ---------- 画图: 3x2 诊断面板 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
ax = axes[0][0]
ax.plot(rev.index, rev, marker="o", ms=3)
ax.set_title("原始周度销售额")
ax.tick_params(axis="x", rotation=45)

ax = axes[0][1]  # ACF
plot_acf(rev, lags=nlag, ax=ax, alpha=0.05)
ax.set_title("自相关函数 ACF")

ax = axes[1][0]  # 差分后
plot_acf(rev.diff().dropna(), lags=nlag, ax=ax, alpha=0.05)
ax.set_title("一阶差分后 ACF")

ax = axes[1][1]  # 季节分解
try:
    decomp_res = decomp.resid
    ax.plot(decomp.trend.index, decomp.trend, lw=2, label="趋势")
    ax.plot(decomp.seasonal.index, decomp.seasonal, lw=1.5, label="季节(周期12周)")
    ax.plot(decomp.resid.index, decomp.resid, alpha=0.6, label="残差")
    ax.legend()
    ax.set_title("STL 季节分解 (趋势/季节/残差)")
except Exception:
    ax.text(0.5, 0.5, "分解失败", ha="center")
ax.tick_params(axis="x", rotation=45)

plt.suptitle("电商周度销售额 —— 序列诊断 (数理统计)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/diagnosis_series.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\n诊断图已输出: {OUT}/diagnosis_series.png")
