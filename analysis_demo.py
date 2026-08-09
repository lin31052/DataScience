"""
DataScience 数据分析 + 中文可视化演示
流程: 造数据 → pandas 清洗聚合 → seaborn 统计图 → matplotlib 中文渲染出图
运行: conda run -n data-analytics python analysis_demo.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 必须在设置中文字体之前调用(否则 seaborn 会重置 rcParams 字体)
sns.set_theme(style="whitegrid", palette="muted")

# ---------- 中文字体自动适配 (CentOS/Ubuntu 通用) ----------
def setup_chinese_font():
    """显式注册中文字体文件, 返回可用族名列表"""
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
    # 收集可用中文字体族
    zh_names = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                "Noto Sans CJK JP", "Noto Sans CJK SC"]
    available = [n for n in zh_names if any(f.name == n for f in fm.fontManager.ttflist)]
    if not available:
        available = sorted({f.name for f in fm.fontManager.ttflist
                            if any(k in f.name for k in ("CJK", "WenQuanYi"))})
    return available

zh_fonts = setup_chinese_font()
print("可用中文字体:", zh_fonts if zh_fonts else "无(将用默认字体)")
if zh_fonts:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = zh_fonts + ["DejaVu Sans"]
    # 关键: 同时设置 font.family 直接指向第一个可用字体
    fm.findfont(fm.FontProperties(family=zh_fonts[0]))  # 触发缓存
plt.rcParams["axes.unicode_minus"] = False  # 负号显示

# ---------- 1. 造数据 ----------
np.random.seed(2026)
n = 240
df = pd.DataFrame({
    "日期": pd.date_range("2025-01-01", periods=n, freq="D"),
    "销售额": np.round(np.random.lognormal(mean=4.2, sigma=0.45, size=n), 2),
    "订单量": np.random.poisson(lam=45, size=n).astype(int),
    "渠道": np.random.choice(["线上", "线下", "分销"], size=n, p=[0.5, 0.3, 0.2]),
    "客单价": np.round(np.random.uniform(60, 380, size=n), 2),
})

# ---------- 2. pandas 分析 ----------
print("=" * 50)
print("数据维度:", df.shape)
print("\n缺失值:\n", df.isnull().sum().to_string())
print("\n描述统计:\n", df.describe().round(2).to_string())

df["月"] = df["日期"].dt.month
agg = df.groupby("渠道").agg(
    销售总额=("销售额", "sum"),
    平均单量=("订单量", "mean"),
    占比=("销售额", lambda s: round(100 * s.sum() / df["销售额"].sum(), 1)),
).round(2)
print("\n按渠道聚合:\n", agg.to_string())

monthly = df.groupby("月")["销售额"].sum()
print("\n月度销售趋势:\n", monthly.round(2).to_string())

# ---------- 3. 可视化 (中文) ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 渠道占比
axes[0, 0].pie(agg["销售总额"], labels=agg.index, autopct="%1.1f%%",
               startangle=90, colors=["#4C72B0", "#55A868", "#C44E52"])
axes[0, 0].set_title("各渠道销售占比")

# 客单价 vs 销售额
sns.scatterplot(data=df, x="客单价", y="销售额", hue="渠道", alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title("客单价与销售额关系")

# 订单量箱线图
sns.boxplot(data=df, x="渠道", y="订单量", ax=axes[1, 0])
axes[1, 0].set_title("各渠道订单量分布")

# 月度趋势
axes[1, 1].plot(monthly.index, monthly.values, marker="o", color="#4C72B0", linewidth=2)
axes[1, 1].set_title("月度销售额趋势")
axes[1, 1].set_xlabel("月份")
axes[1, 1].set_ylabel("销售额")

plt.tight_layout()
out = "/tmp/datascience_chart.png"
plt.savefig(out, dpi=120)
print(f"\n✅ 中文图表已保存: {out}")
print("\n" + "=" * 50)
print("数据分析 + 中文可视化全流程成功!")
print("ALL_OK")
