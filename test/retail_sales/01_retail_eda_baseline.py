"""
电商零售销售预测 Step 1 — 数据准备 + EDA + 基线模型
数据: UCI Online Retail (541909 行交易, 英国线上礼品店 2010-12~2011-12)
任务: 预测周度销售额 (时间序列, 53 周)
依赖: pandas sklearn matplotlib seaborn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------- 中文字体 (必须在 set_theme 之后设置) ----------
sns.set_theme(style="whitegrid", palette="muted")

def setup_chinese_font():
    import glob
    font_files = []
    for pat in [
        "/usr/share/fonts/**/wqy-microhei.ttc",
        "/usr/share/fonts/**/wqy-zenhei.ttc",
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/*CJK*.ttc",
        "/usr/share/fonts/**/*WenQuanYi*",
    ]:
        font_files += glob.glob(pat, recursive=True)
    for fp in sorted(set(font_files)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    zh_names = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                "Noto Sans CJK JP", "Noto Sans CJK SC"]
    available = [n for n in zh_names if any(f.name == n for f in fm.fontManager.ttflist)]
    if not available:
        available = sorted({f.name for f in fm.fontManager.ttflist
                            if any(k in f.name for k in ("CJK", "WenQuanYi"))})
    return available

zh_fonts = setup_chinese_font()
print("可用中文字体:", zh_fonts if zh_fonts else "无(默认)")
if zh_fonts:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = zh_fonts + ["DejaVu Sans"]
    fm.findfont(fm.FontProperties(family=zh_fonts[0]))
plt.rcParams["axes.unicode_minus"] = False

import os
OUT = os.getcwd() + "/test/retail_sales/output"
os.makedirs(OUT, exist_ok=True)
CSV = os.getcwd() + "/test/retail_sales/online_retail.csv"

# ---------- 1. 读数据 ----------
df = pd.read_csv(CSV, parse_dates=["InvoiceDate"])
print("=" * 60)
print("UCI Online Retail 电商交易销售预测")
print("=" * 60)
print(f"原始交易数: {len(df)}, 字段: {list(df.columns)}")

# 仅保留英国市场 (占 91%)
df_uk = df[df["Country"] == "United Kingdom"].copy()
print(f"英国市场交易: {len(df_uk)} ({len(df_uk)/len(df)*100:.1f}%)")

# 清洗: 删缺客户ID、负数量(退货)、退单(InvoiceNo 以 C 开头)
before = len(df_uk)
df_uk = df_uk[df_uk["CustomerID"].notna()]
df_uk = df_uk[df_uk["Quantity"] > 0]
df_uk = df_uk[~df_uk["InvoiceNo"].astype(str).str.startswith("C")]
print(f"清洗后: {len(df_uk)} ({len(df_uk)/before*100:.1f}%)")

# 单行销售额
df_uk["Revenue"] = df_uk["Quantity"] * df_uk["UnitPrice"]

# ---------- 2. 周度聚合 ----------
df_uk["Week"] = df_uk["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time)
weekly = df_uk.groupby("Week").agg(
    Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Items=("Quantity", "sum"),
).sort_index()
print(f"\n周度聚合: {len(weekly)} 周")
print(f"周销售额范围: £{weekly['Revenue'].min():,.0f} ~ £{weekly['Revenue'].max():,.0f}")
print(f"周销售额均值: £{weekly['Revenue'].mean():,.0f}")

# ---------- 3. EDA 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
ax = axes[0][0]
ax.plot(weekly.index, weekly["Revenue"], marker="o", ms=3, lw=1.5)
ax.set_title("周度销售额走势")
ax.set_ylabel("销售额 (£)")
ax.tick_params(axis="x", rotation=45)

ax = axes[0][1]
ax.bar(weekly.index, weekly["Orders"], color="steelblue", width=6)
ax.set_title("周度订单量")
ax.tick_params(axis="x", rotation=45)

ax = axes[1][0]
sns.histplot(df_uk["Revenue"], bins=50, ax=ax, color="coral")
ax.set_title("单笔交易销售额分布 (右偏)")
ax.set_xlabel("单笔金额 (£)")

ax = axes[1][1]
top_country = df_uk.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=top_country.head(5).values, y=top_country.head(5).index, ax=ax, palette="viridis")
ax.set_title("Top5 国家销售额 (英国本身最高)")
ax.set_xlabel("总销售额 (£)")

plt.suptitle("电商零售数据 EDA (UCI Online Retail)", fontsize=15, y=1.02)
plt.tight_layout()
eda_png = f"{OUT}/eda_weekly.png"
plt.savefig(eda_png, dpi=100, bbox_inches="tight")
plt.close()

# ---------- 4. 基线时序模型: 日期特征 + 滞后 ----------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

ts = weekly["Revenue"].copy()
ts.index.name = "week_start"

# 特征: 周序号_ 月份/季度周/滞后1-2周
feat = pd.DataFrame(index=ts.index)
feat["week_num"] = range(len(ts))
feat["month"] = ts.index.month
feat["quarter"] = ts.index.quarter
feat["lag1"] = ts.shift(1)
feat["lag2"] = ts.shift(2)
feat["target"] = ts

# 滚动均值 (不含当前, 前4周)
feat["roll_mean4"] = ts.shift(1).rolling(4).mean()

feat = feat.dropna()
X = feat.drop(columns=["target"])
y = feat["target"]

# 时间切分: 后 20% 作测试
split = int(len(X) * 0.8)
X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y[:split], y[split:]
print(f"\n训练: {len(X_tr)} 周, 测试: {len(X_te)} 周")

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
pred = rf.predict(X_te)

rmse = np.sqrt(mean_squared_error(y_te, pred))
mae = mean_absolute_error(y_te, pred)
y_base = np.full_like(pred, y_tr.mean())
rmse_base = np.sqrt(mean_squared_error(y_te, y_base))
print(f"\n[基线] 随机森林(lag+日期): RMSE=£{rmse:,.0f}  MAE=£{mae:,.0f}")
print(f"[对比] 均值基线(预测=均值):   RMSE=£{rmse_base:,.0f}")
print(f"改进: RMSE 降低 {((rmse_base-rmse)/rmse_base*100):.0f}%")
print("特征重要性:", dict(zip(X.columns, np.round(rf.feature_importances_, 3))))

# 预测图
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_tr.index, y_tr, color="steelblue", lw=2, label="训练 (真实)")
ax.plot(y_te.index, y_te, color="green", lw=2, label="测试 (真实)")
ax.plot(y_te.index, pred, color="red", lw=2, ls="--", label="预测 (随机森林)")
ax.axvline(y_te.index[0], color="gray", ls=":", label="训练/测试分界")
ax.set_title(f"电商周度销售额预测 (基线 RF, 测试RMSE=£{rmse:,.0f})")
ax.set_ylabel("销售额 (£)")
ax.legend()
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_baseline.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n图表已输出到:", OUT)
