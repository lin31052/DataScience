"""
电商零售销售预测 Step 2 — 算法演进: 更强特征 + 多模型对比 + 多任务
数据: UCI Online Retail (清洗后 UK 35万交易 → 聚合 53 周)
任务A: 周度销售额预测 (时间序列回归)
任务B: 周度订单量预测 (换任务视角)
模型: LinearRegression / RandomForest / XGBoost 三选对比
特征: lag1-4 + 滚动均值(4/12周) + 滚动标准差(4) + 差分 + 日期特征
目标: 相对 step1 基线(RF=£69,692) 进一步压低 RMSE
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- 读 + 清洗 (与 step1 相同) ----------
CSV = os.getcwd() + "/test/retail_sales/online_retail.csv"
df = pd.read_csv(CSV, parse_dates=["InvoiceDate"])
df_uk = df[df["Country"] == "United Kingdom"].copy()
df_uk = df_uk[df_uk["CustomerID"].notna()]
df_uk = df_uk[df_uk["Quantity"] > 0]
df_uk = df_uk[~df_uk["InvoiceNo"].astype(str).str.startswith("C")]
df_uk["Revenue"] = df_uk["Quantity"] * df_uk["UnitPrice"]

# ---------- 周度聚合: 销售额 + 订单量 双目标 ----------
df_uk["Week"] = df_uk["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time)
wk = df_uk.groupby("Week").agg(
    Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique"),
).sort_index()
print(f"周度序列: {len(wk)} 周")

# ---------- 特征工程 (升级版) ----------
def build_features(series):
    f = pd.DataFrame(index=series.index)
    f["week_num"] = range(len(series))
    f["month"] = series.index.month
    f["quarter"] = series.index.quarter
    # lag 1-4
    for k in (1, 2, 3, 4):
        f[f"lag{k}"] = series.shift(k)
    # 滚动统计 (不含当期)
    f["roll_mean4"] = series.shift(1).rolling(4).mean()
    f["roll_mean12"] = series.shift(1).rolling(12).mean()
    f["roll_std4"] = series.shift(1).rolling(4).std()
    # 同比前4周同周均值 (季节)
    f["roll_std4"] = f["roll_std4"].fillna(f["roll_std4"].mean())
    # 差分 (必须只用过去信息, shift后再diff避免泄漏当期值)
    f["diff1"] = series.shift(1).diff(1)
    # 前一年同周 (周 53 太少, 用前4周均值近似季节)
    return f

def evaluate(X, y, split, model, name):
    split = max(int(len(X) * split), len(X) - 10)  # 至少留10周测试
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y[:split], y[split:]
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    base = np.full_like(pred, y_tr.mean())
    rmse_base = np.sqrt(mean_squared_error(y_te, base))
    return {
        "任务目标": name,
        "模型": type(model).__name__,
        "训练周数": len(X_tr), "测试周数": len(X_te),
        "RMSE": round(rmse), "MAE": round(mae), "R²": round(r2, 3),
        "均值基线RMSE": round(rmse_base),
        "改进%": round((rmse_base - rmse) / rmse_base * 100),
        "pred": pred, "y_te": y_te, "y_tr": y_tr,
        "idx_tr": list(X.index[:split]), "idx_te": list(X.index[split:]),
    }

# ---------- 模型工厂 ----------
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = [
    ("线性回归", LinearRegression()),
    ("随机森林", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("XGBoost", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                              random_state=42, n_jobs=-1, verbosity=0)),
]

results = []

# ===== 任务A: 销售额 (7:3 切分全部53周) =====
print("\n" + "=" * 62)
print("任务A: 周度销售额预测 (£)")
print("=" * 62)
Xs = build_features(wk["Revenue"])
ys = wk["Revenue"]
feats_all = build_features(wk["Revenue"])
feats_all["target"] = ys
feats_all = feats_all.dropna()
XA = feats_all.drop(columns=["target"]); yA = feats_all["target"]
# 复制原始时序给画图
da = {}
for name, m in models:
    r = evaluate(XA.copy(), yA.copy(), 0.75, m, "销售额")
    results.append(r)
    da[type(m).__name__] = r
    print(f"  {type(m).__name__:<8} RMSE=£{r['RMSE']:>8,}  MAE=£{r['MAE']:>7,}  R²={r['R²']}  改进={r['改进%']}%")

# ===== 任务B: 订单量 =====
print("\n" + "=" * 62)
print("任务B: 周度订单量预测 (单)  均值基线 RMSE 参考下方")
print("=" * 62)
featsB = build_features(wk["Orders"])
featsB["target"] = wk["Orders"]
featsB = featsB.dropna()
XB = featsB.drop(columns=["target"]); yB = featsB["target"]
db = {}
for name, m in models:
    r = evaluate(XB.copy(), yB.copy(), 0.75, m, "订单量")
    results.append(r)
    db[type(m).__name__] = r
    print(f"  {type(m).__name__:<8} RMSE={r['RMSE']:>6,}单  MAE={r['MAE']:>5,}  R²={r['R²']}  改进={r['改进%']}%")

# ---------- 画图: 每个任务取最优模型 ----------
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

def plot_task(results_map, title, ylab, fname):
    """用各模型返回的对齐时间索引画 真实vs各模型预测(测试段)"""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    # 取第一个模型的训练/测试索引
    r0 = list(results_map.values())[0]
    idx_tr = r0["idx_tr"]; idx_te = r0["idx_te"]
    # 完整真实序列 (训练+测试)
    ytr_ts = pd.Series(r0["y_tr"], index=idx_tr)
    yte_ts = pd.Series(r0["y_te"], index=idx_te)
    ax.plot(idx_tr, ytr_ts, color="steelblue", lw=2, label="训练 (真实)")
    ax.plot(idx_te, yte_ts, color="green", lw=2, label="测试 (真实)")
    colors = {"线性回归": "teal", "随机森林": "orange", "XGBoost": "crimson"}
    for nm, r in results_map.items():
        ax.plot(r["idx_te"], r["pred"], ls="--", lw=1.8,
                label=f'{nm} (RMSE=£{r["RMSE"]:,})')
    ax.axvline(idx_te[0], color="gray", ls=":", label="训练/测试分界")
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=100, bbox_inches="tight")
    plt.close()

# 画图 (用各模型返回的时间索引)
plot_task(da, "电商周度销售额预测: 三模型对比", "销售额 (£)", "forecast_multi_revenue.png")
plot_task(db, "电商周度订单量预测: 三模型对比", "订单量 (单)", "forecast_multi_orders.png")

# ---------- 汇总表 ----------
print("\n" + "=" * 62)
print("汇总 (改进% = 相对该任务均值基线的 RMSE 下降)")
print("=" * 62)
import json
summary = [{"任务": r["任务目标"], "模型": r["模型"], "RMSE": r["RMSE"],
            "MAE": r["MAE"], "R2": r["R²"], "均值基线RMSE": r["均值基线RMSE"],
            "改进%": r["改进%"]} for r in results]
print(pd.DataFrame(summary).to_string(index=False))
print("\n图表已输出到:", OUT)
