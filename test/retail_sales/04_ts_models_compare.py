"""
电商零售预测 Step 4 — 严格时序模型对比 (walk-forward 滚动验证)
关键洞察(来自 step3): Naive(上周值) RMSE=£46k << 我们RF的 £76k!
  说明: ①序列非平稳有趋势 ②强自相关 ③真正信号在 lag 里, 我们之前堆错了特征
策略: 用真正的时序模型(ETS/ARIMA) 显式建模趋势+自回归, 全部打 Naive 基准,
      并采用 rolling walk-forward 多折评估(比单次切分稳健)。
数据: UK 53 周销售额。
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

# ---------- 读+清洗+周度聚合 (与前面一致) ----------
df = pd.read_csv(CSV, parse_dates=["InvoiceDate"])
df_uk = df[df["Country"]=="United Kingdom"].copy()
df_uk = df_uk[df_uk["CustomerID"].notna()]
df_uk = df_uk[df_uk["Quantity"]>0]
df_uk = df_uk[~df_uk["InvoiceNo"].astype(str).str.startswith("C")]
df_uk["Revenue"] = df_uk["Quantity"]*df_uk["UnitPrice"]
df_uk["Week"] = df_uk["InvoiceDate"].dt.to_period("W").apply(lambda r: r.start_time)
wk = df_uk.groupby("Week").agg(Revenue=("Revenue","sum")).sort_index()
rev = wk["Revenue"].astype(float)

print("=" * 64)
print("电商周度销售额 — 严格时序模型对比 (walk-forward)")
print("=" * 64)
print(f"序列长度: {len(rev)} 周")

# ---------- 统一 walk-forward 评估: 训练窗起点=0, 每折预测连续 h 周 ----------
def walk_forward_predict(make_forecast, horizon=6, min_train=25):
    """make_forecast(y_train, n_future) -> 返回未来n周的预测数组。
    从 min_train 起, 每折训练窗口扩展 min_train + k*horizon 周, 预测其后 horizon 周。"""
    n = len(rev.values)
    preds = {}   # 周index -> 预测值
    starts = list(range(min_train, n - horizon + 1, horizon))
    for start in starts:
        tr = rev.iloc[:start]
        nf = min(horizon, n - start)
        f = make_forecast(tr.values, nf)
        for j in range(nf):
            preds[start + j] = f[j]
    # 建对齐数组
    idx_all = np.arange(n)
    pred_arr = np.full(n, np.nan)
    real_arr = rev.values
    for k, v in preds.items():
        pred_arr[k] = v
    # 只评估被预测到的点
    mask = ~np.isnan(pred_arr)
    return real_arr[mask], pred_arr[mask], mask

# ---------- 模型定义 (全部返回未来 n 周预测) ----------
def f_naive(tr, n):
    return np.full(n, tr[-1])

def f_drift(tr, n):
    """漂移: 线性趋势外推, 每步加训练期平均差"""
    if len(tr) < 2:
        return np.full(n, tr[-1] if len(tr) > 0 else 0)
    slope = (tr[-1] - tr[0]) / (len(tr) - 1)
    return np.array([tr[-1] + slope * (i + 1) for i in range(n)])

from statsmodels.tsa.holtwinters import ExponentialSmoothing
def f_holt(tr, n):
    try:
        m = ExponentialSmoothing(tr, trend="add", damped_trend=False,
                                 initialization_method="estimated").fit(optimized=True)
        return m.forecast(n)
    except Exception:
        return np.full(n, tr[-1])

def f_ses(tr, n):
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        m = SimpleExpSmoothing(pd.Series(tr)).fit(optimized=True)
        return np.array(m.forecast(n))
    except Exception:
        return np.full(n, tr[-1])

from statsmodels.tsa.arima.model import ARIMA
def f_arima(tr, n):
    try:
        m = ARIMA(tr, order=(1,1,1)).fit()
        return np.array(m.forecast(n))
    except Exception:
        return np.full(n, tr[-1])

from statsmodels.tsa.arima.model import ARIMA as ARIMA2
def f_arma_d1(tr, n):
    """仅差分1后再用lag结构(AR(1) on diff)"""
    try:
        d = np.diff(tr)
        m = ARIMA2(d, order=(1,0,1)).fit()
        fc = np.array(m.forecast(n))
        # 反差分
        out = [tr[-1]]
        for x in fc:
            out.append(out[-1] + x)
        return np.array(out[1:])
    except Exception:
        return np.full(n, tr[-1])

# ---------- 运行对比 ----------
horizon = 6
models = {
    "Naive(上周值)": f_naive,
    "Drift(趋势漂移)": f_drift,
    "SES指数平滑": f_ses,
    "Holt线性趋势": f_holt,
    "ARIMA(1,1,1)": f_arima,
    "Diff图+ARMA(差分建模)": f_arma_d1,
}

from sklearn.metrics import mean_absolute_error, mean_squared_error
scores = []
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(rev.index, rev, color="black", lw=2, label="真实")

all_forecast_mask = None
line_styles = {"Naive(上周值)": ("--", "gray"), "Drift(趋势漂移)": ("-.", "teal"),
               "SES指数平滑": (":", "orange"), "Holt线性趋势": ("--", "purple"),
               "ARIMA(1,1,1)": ("--", "crimson"), "Diff图+ARMA(差分建模)": ("--", "green")}

for name, fn in models.items():
    real_test, pred_test, mask = walk_forward_predict(fn, horizon=horizon)
    mae = mean_absolute_error(real_test, pred_test)
    rmse = np.sqrt(mean_squared_error(real_test, pred_test))
    # 均值基线(在真值上)
    mean_base_rmse = np.sqrt(mean_squared_error(real_test, np.full_like(real_test, rev.mean())))
    impr = (mean_base_rmse - rmse) / mean_base_rmse * 100
    scores.append({"模型": name, "MAE": round(mae), "RMSE": round(rmse),
                   "均值基线RMSE": round(mean_base_rmse), "改进%": round(impr)})
    # 画 forecast
    ls, c = line_styles.get(name, ("--", "blue"))
    xs = rev.index[mask]
    ax.plot(xs, pred_test, ls=ls, color=c, lw=1.5, label=f'{name} (RMSE=£{rmse:,})')

scores_df = pd.DataFrame(scores).sort_values("RMSE")
print("\n【walk-forward 多折评估】(每折训练窗扩展, 预测未来6周)")
print(scores_df.to_string(index=False))

best = scores_df.iloc[0]["模型"]
print(f"\n🏆 最优: {best}")

ax.set_title("电商周度销售额时序预测: 严格时序模型对比 (walk-forward)")
ax.set_ylabel("销售额 (£)")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_ts_models.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"\n对比图已输出: {OUT}/forecast_ts_models.png")
