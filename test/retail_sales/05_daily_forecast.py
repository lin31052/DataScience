"""
电商零售预测 Step 5 — 细化到日度粒度: 星期效应 + 7天滞后结构
动机: 周度仅53点, 序列固有噪声大, 时序方法封顶 RMSE≈42k(Naive/Drift)。
      日度有 ~340 点, 可捕捉星期几效应(周末/工作日差异), 数据增6倍。
策略: 日度销售额, 特征含 [星期几独热 + lag1/lag7/lag14 + 滚动7日均值],
      用 walk-forward 对比 Naive/Drift/RF/XGBoost。
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

def setup_chinese_font():
    import glob
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

# ---------- 读+清洗+日度聚合 ----------
df = pd.read_csv(CSV, parse_dates=["InvoiceDate"])
df_uk = df[df["Country"]=="United Kingdom"].copy()
df_uk = df_uk[df_uk["CustomerID"].notna()]
df_uk = df_uk[df_uk["Quantity"]>0]
df_uk = df_uk[~df_uk["InvoiceNo"].astype(str).str.startswith("C")]
df_uk["Revenue"] = df_uk["Quantity"]*df_uk["UnitPrice"]
df_uk["Day"] = df_uk["InvoiceDate"].dt.floor("D")
day = df_uk.groupby("Day").agg(Revenue=("Revenue","sum")).sort_index()
rev = day["Revenue"].astype(float)

print("=" * 64)
print("电商日度销售额预测 (细化到天, 星期效应+7天滞后)")
print("=" * 64)
print(f"日度序列长度: {len(rev)} 天 ({rev.index.min().date()} ~ {rev.index.max().date()})")
print(f"日销售额均值={rev.mean():,.0f}  最大值={rev.max():,.0f}")

# ---------- 星期几分析 ----------
dow_stat = rev.groupby(rev.index.dayofweek).mean()
dow_names = ["周一","周二","周三","周四","周五","周六","周日"]
print("\n【星期几平均销售额】(揭示周期) ")
for i in range(7):
    print(f"  {dow_names[i]}: £{dow_stat.iloc[i]:,.0f}")

# ---------- 构造特征 ----------
f = pd.DataFrame(index=rev.index)
f["target"] = rev
# 星期几 独热
for d in range(7):
    f[dow_names[d]] = (rev.index.dayofweek == d).astype(int)
# 7天滞后 (周周期)
for k in (1, 2, 3, 7, 14, 21):
    f[f"lag{k}"] = rev.shift(k)
# 过去7天滚动均值 (不含当天)
f["roll7"] = rev.shift(1).rolling(7).mean()
# 月度内日序号
f["dom"] = rev.index.day
f = f.dropna()
X = f.drop(columns=["target"]); y = f["target"]

# ---------- walk-forward 评估 ----------
def walk_forward_model(model_fn, n_future=14, min_train=60):
    """model_fn(X, y, n_future) 返回未来n_future的预测数组 (在完整 X 上的测试窗口)"""
    n = len(y)
    preds = {}
    for start in range(min_train, n - n_future + 1, n_future):
        Xtr, ytr = X.iloc[:start], y.iloc[:start]
        nf = min(n_future, n - start)
        Xte_next = X.iloc[start:start+nf]
        f = model_fn(Xtr, ytr, Xte_next)
        if len(f) != nf:
            f = list(f)[:nf] + [ytr.mean()]*(nf-len(f))
        for j in range(nf):
            preds[start+j] = f[j]
    pred_arr = np.full(n, np.nan)
    for k,v in preds.items():
        pred_arr[k] = v
    mask = ~np.isnan(pred_arr)
    return y.values[mask], pred_arr[mask], mask

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def f_naive(Xtr, ytr, Xte):
    return np.full(len(Xte), ytr.iloc[-1])

def f_drift(Xtr, ytr, Xte):
    n = len(Xte); L = len(ytr)
    slope = (ytr.iloc[-1]-ytr.iloc[0])/max(L-1,1)
    return np.array([ytr.iloc[-1]+slope*(i+1) for i in range(n)])

def f_rf(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)

def f_xgb(Xtr, ytr, Xte):
    m = XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     n_jobs=-1, verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)

models = {"Naive(昨/上周)": f_naive, "Drift(趋势)": f_drift,
          "随机森林(日+lag7+星期)": f_rf, "XGBoost(日+lag7+星期)": f_xgb}

print("\n【walk-forward 日度预测】(训练窗扩展, 预测未来14天)")
scores = []
for name, fn in models.items():
    rt, pt, mk = walk_forward_model(fn, n_future=14)
    mae = mean_absolute_error(rt, pt)
    rmse = np.sqrt(mean_squared_error(rt, pt))
    mbase = np.sqrt(mean_squared_error(rt, np.full_like(rt, rev.mean())))
    impr = (mbase-rmse)/mbase*100
    scores.append({"模型": name, "MAE": round(mae), "RMSE": round(rmse),
                   "均值基线RMSE": round(mbase), "改进%": round(impr)})
    print(f"  {name:<26} MAE=£{mae:>8,.0f}  RMSE=£{rmse:>8,.0f}  改进={impr:>3}%")

scores_df = pd.DataFrame(scores).sort_values("RMSE")
print("\n" + scores_df.to_string(index=False))
print(f"\n🏆 日度最优: {scores_df.iloc[0]['模型']} (RMSE=£{scores_df.iloc[0]['RMSE']:,})")

# ---------- 画图: 测试段某个窗口 ----------
fig, ax = plt.subplots(figsize=(13, 5.5))
# 选最后一段做展示
n_future = 14
start_show = len(rev) - n_future
tr = rev.iloc[:start_show]; te = rev.iloc[start_show:]
colors = {"Naive(昨/上周)": "gray", "Drift(趋势)": "teal",
          "随机森林(日+lag7+星期)": "orange", "XGBoost(日+lag7+星期)": "crimson"}
ax.plot(tr.index, tr.values, color="black", lw=2, label="训练(真实)")
ax.plot(te.index, te.values, color="green", lw=2, label="测试(真实)")
for name, fn in models.items():
    # 要用最后段的训练X
    Xtr, ytr, Xte_t = X.iloc[:start_show], y.iloc[:start_show], X.iloc[start_show:]
    f = fn(Xtr, ytr, Xte_t)
    ax.plot(te.index[:len(f)], f, ls="--", lw=1.5, color=colors[name], label=name)
ax.set_title("电商日度销售额预测: Naive/Drift/随机森林/XGBoost 对比 (日粒度)")
ax.set_ylabel("日销售额 (£)")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_daily_compare.png", dpi=100, bbox_inches="tight")
plt.close()

# 星期效应图
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(x=dow_names, y=dow_stat.values, palette="viridis", ax=ax2)
ax2.set_title("星期几 平均销售额 (规律性)")
ax2.set_ylabel("平均销售额 (£)")
plt.tight_layout()
plt.savefig(f"{OUT}/dow_effect.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\n图表已输出: {OUT}/forecast_daily_compare.png, {OUT}/dow_effect.png")
