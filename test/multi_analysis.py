"""
经典数据集核心分析 (多数据集版)
覆盖: 波士顿房价 / 葡萄酒 / 糖尿病 / 加州房价
每个数据集: 概览 → 描述统计 → 目标变量分析 → 中文可视化
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---------- 中文字体 (必须在 set_theme 之后) ----------
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

# ======================================================
# 1. 波士顿房价 (回归, 目标 medv)
# ======================================================
print("=" * 56)
print("1. 波士顿房价数据集 (506 样本, 回归预测 medv)")
print("=" * 56)
boston = pd.read_csv("test/boston/boston.csv")
print(f"形状: {boston.shape}")
print(f"目标 medv 房价均值: ${boston['medv'].mean():.0f}k, 中位数: ${boston['medv'].median():.0f}k")
print(f"范围: ${boston['medv'].min():.0f}k ~ ${boston['medv'].max():.0f}k")
print(f"缺失值: {boston.isnull().sum().sum()}")
# 关键特征相关性
corr = boston.corr()["medv"].drop("medv").sort_values(key=abs, ascending=False)
print("\n与房价最相关的前5个特征:")
for feat, v in corr.head(5).items():
    print(f"  {feat:8s} 相关={v:+.3f}")
print(f"\n特征数量: {boston.shape[1]-1} 个")
print("特征: crim(犯罪率) rm(房间数) lstat(低收入) tax(税率) age(房龄) 等")

# ======================================================
# 2. 葡萄酒 (分类, 目标 target 0/1/2)
# ======================================================
print("\n" + "=" * 56)
print("2. 葡萄酒数据集 (178 样本, 3 类分类)")
print("=" * 56)
wine = pd.read_csv("test/wine/wine.csv")
print(f"形状: {wine.shape}")
print("类别分布:", wine["target"].value_counts().sort_index().to_dict())
print(f"缺失值: {wine.isnull().sum().sum()}")
print(f"特征数: {wine.shape[1]-1} 个 (酒精/苹果酸/灰分/类黄酮 等)")
corr2 = wine.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
print("\n与类别最相关的前5个特征:")
for feat, v in corr2.head(5).items():
    print(f"  {feat:22s} 相关={v:+.3f}")

# ======================================================
# 3. 糖尿病 (回归, 目标 target 病情进展)
# ======================================================
print("\n" + "=" * 56)
print("3. 糖尿病数据集 (442 样本, 回归预测)")
print("=" * 56)
dia = pd.read_csv("test/diabetes/diabetes.csv")
print(f"形状: {dia.shape}")
print(f"目标 target 统计: 均值={dia['target'].mean():.0f}, 范围={dia['target'].min():.0f}~{dia['target'].max():.0f}")
print(f"缺失值: {dia.isnull().sum().sum()}")
print("特征: age/bmi/bp/s1-s6 (标准化后的生理指标)")
corr3 = dia.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
print("\n与目标最相关的前5个特征:")
for feat, v in corr3.head(5).items():
    print(f"  {feat:7s} 相关={v:+.3f}")

# ======================================================
# 4. 加州房价 (回归, 目标 median_house_value)
# ======================================================
print("\n" + "=" * 56)
print("4. 加州房价数据集 (20640 样本, 回归预测)")
print("=" * 56)
cal = pd.read_csv("test/housing.csv")
print(f"形状: {cal.shape}")
print(f"目标 median_house_value: 均值=${cal['median_house_value'].mean():.0f}, 中位=${cal['median_house_value'].median():.0f}")
print(f"缺失值:\n{cal.isnull().sum()[cal.isnull().sum()>0].to_string() if (cal.isnull().sum()>0).any() else '无'}")
print("海岸线分布:", cal["ocean_proximity"].value_counts().to_dict())
corr4 = cal.select_dtypes("number").corr()["median_house_value"].drop("median_house_value").sort_values(key=abs, ascending=False)
print("\n与房价最相关的前5个特征:")
for feat, v in corr4.head(5).items():
    print(f"  {feat:16s} 相关={v:+.3f}")

# ======================================================
# 可视化 (4 数据集 各一图)
# ======================================================
print("\n" + "=" * 56)
print("生成可视化...")
print("=" * 56)

# 图1: 波士顿 —— 房间数 vs 房价 + 低收入占比 vs 房价
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=boston, x="rm", y="medv", alpha=0.5, ax=axes[0])
axes[0].set_title("波士顿房价: 房间数 vs 价格")
axes[0].set_xlabel("平均房间数 rm")
axes[0].set_ylabel("房价(千美元)")
sns.scatterplot(data=boston, x="lstat", y="medv", alpha=0.5, color="#C44E52", ax=axes[1])
axes[1].set_title("波士顿房价: 低收入占比 vs 价格")
axes[1].set_xlabel("低收入人群占比 lstat(%)")
axes[1].set_ylabel("房价(千美元)")
plt.tight_layout()
plt.savefig("/tmp/boston_chart.png", dpi=120)
print("✅ 波士顿图: /tmp/boston_chart.png")

# 图2: 葡萄酒 —— 特征相关性热力图 + 两特征散点
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top3 = corr2.head(3).index.tolist()
sns.scatterplot(data=wine, x=top3[0], y=top3[1], hue="target", palette="deep", ax=axes[0])
axes[0].set_title(f"葡萄酒: {top3[0]} vs {top3[1]}(按类别)")
sns.heatmap(wine.corr(), cmap="RdBu_r", center=0, ax=axes[1], cbar_kws={"shrink": 0.8})
axes[1].set_title("葡萄酒特征相关性热力图")
plt.tight_layout()
plt.savefig("/tmp/wine_chart.png", dpi=120)
print("✅ 葡萄酒图: /tmp/wine_chart.png")

# 图3: 糖尿病 —— bmi vs target + 特征相关性条形
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=dia, x="bmi", y="target", alpha=0.5, ax=axes[0])
axes[0].set_title("糖尿病: BMI vs 病情进展")
axes[0].set_xlabel("BMI(标准化)")
axes[0].set_ylabel("病情进展")
imp = corr3.head(6)
sns.barplot(x=imp.values, y=imp.index, palette="muted", ax=axes[1])
axes[1].set_title("糖尿病: 与目标最相关特征")
axes[1].set_xlabel("相关系数")
plt.tight_layout()
plt.savefig("/tmp/diabetes_chart.png", dpi=120)
print("✅ 糖尿病图: /tmp/diabetes_chart.png")

# 图4: 加州房价 —— 收入 vs 房价 + 地理分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=cal.sample(3000, random_state=42), x="median_income", y="median_house_value",
                alpha=0.3, s=10, ax=axes[0])
axes[0].set_title("加州房价: 中位收入 vs 房价")
axes[0].set_xlabel("中位收入(万美元)")
axes[0].set_ylabel("房价(美元)")
sns.scatterplot(data=cal.sample(5000, random_state=42), x="longitude", y="latitude",
                hue="ocean_proximity", s=5, alpha=0.6, ax=axes[1])
axes[1].set_title("加州房价: 地理分布")
axes[1].set_xlabel("经度")
axes[1].set_ylabel("纬度")
plt.tight_layout()
plt.savefig("/tmp/california_chart.png", dpi=120)
print("✅ 加州房价图: /tmp/california_chart.png")

print("\n" + "=" * 56)
print("4 个数据集分析全部完成!")
print("ALL_DATASETS_OK")
