"""
鸢尾花 (Iris) 核心数据分析
数据: test/iris/iris.csv (150 样本, 3 类, 4 特征)
分析: 类别分布 / 特征统计 / 类别区分度 / 特征相关性 / 分类关键结论
输出: 控制台统计 + 中文可视化 /tmp/iris_chart.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---------- 中文字体 (set_theme 先, 字体后) ----------
sns.set_theme(style="whitegrid", palette="muted")
def setup_chinese_font():
    import glob
    font_files = []
    for pat in ["/usr/share/fonts/**/wqy-microhei.ttc", "/usr/share/fonts/**/wqy-zenhei.ttc",
                "/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/*CJK*.ttc"]:
        font_files += glob.glob(pat, recursive=True)
    for fp in sorted(set(font_files)):
        try: fm.fontManager.addfont(fp)
        except Exception: pass
    zh = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "Noto Sans CJK SC"]
    avail = [n for n in zh if any(f.name == n for f in fm.fontManager.ttflist)]
    if not avail:
        avail = sorted({f.name for f in fm.fontManager.ttflist if "CJK" in f.name or "WenQuanYi" in f.name})
    return avail

zh = setup_chinese_font()
print("可用中文字体:", zh)
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    fm.findfont(fm.FontProperties(family=zh[0]))
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 读数据 ----------
df = pd.read_csv("test/iris/iris.csv")
print("=" * 56)
print("鸢尾花 (Iris) 数据集核心分析")
print("=" * 56)
print(f"形状: {df.shape}  ({len(df)} 样本, {df.shape[1]-1} 特征, {df['species'].nunique()} 类)")
print(f"\n类别分布:\n{df['species'].value_counts().to_string()}")
print(f"\n缺失值: {df.isnull().sum().sum()}")

# ---------- 2. 特征统计 ----------
print("\n【各特征描述统计】")
print(df.groupby("species").describe().round(2).to_string())

# ---------- 3. 类别区分度分析 ----------
print("\n【各类别特征均值对比】")
mean_by = df.groupby("species").mean().round(2)
print(mean_by.to_string())

# 特征区分度: 组间方差/组内方差 (越大越能区分)
print("\n【特征区分度】(组间均值差 / 组内标准差, 越大越能区分品种)")
feats = df.columns[:-1]
sep_ratios = {}
for f in feats:
    groups = [df[df["species"] == s][f] for s in df["species"].unique()]
    between = np.std([g.mean() for g in groups])
    within = np.mean([g.std() for g in groups])
    sep_ratios[f] = between / within if within > 0 else 0
for f, v in sorted(sep_ratios.items(), key=lambda x: -x[1]):
    print(f"  {f:22s} 区分度={v:.2f}")

best_feat = max(sep_ratios, key=sep_ratios.get)
print(f"\n🔑 最具区分度的特征: {best_feat} (区分度 {sep_ratios[best_feat]:.2f})")

# ---------- 4. 相关性 ----------
print("\n【特征相关性矩阵】")
corr = df[feats].corr().round(3)
print(corr.to_string())
# 找最高相关对
corr_pairs = []
for i in range(len(feats)):
    for j in range(i+1, len(feats)):
        corr_pairs.append((feats[i], feats[j], corr.iloc[i, j]))
corr_pairs.sort(key=lambda x: -abs(x[2]))
print(f"\n🔑 最相关特征对: {corr_pairs[0][0]} ↔ {corr_pairs[0][1]} (r={corr_pairs[0][2]:.3f})")

# ---------- 5. 可视化 (中文) ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 图1: 特征两两散点(按类别着色) — 看哪个特征能分开品种
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)",
                hue="species", s=60, ax=axes[0, 0])
axes[0, 0].set_title("花萼长 vs 花萼宽 (setosa 明显可分)")

sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)",
                hue="species", s=60, ax=axes[0, 1])
axes[0, 1].set_title("花瓣长 vs 花瓣宽 (三类基本线性可分)")

# 图2: 花瓣长度箱线图(按类别)
sns.boxplot(data=df, x="species", y="petal length (cm)", ax=axes[1, 0])
axes[1, 0].set_title("各类花瓣长度分布")
axes[1, 0].set_xlabel("品种")

# 图3: 特征相关性热力图
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
            ax=axes[1, 1], cbar_kws={"shrink": 0.8})
axes[1, 1].set_title("特征相关性热力图")

plt.tight_layout()
out = "/tmp/iris_chart.png"
plt.savefig(out, dpi=130)
print(f"\n✅ 中文图表已保存: {out}")

# ---------- 6. 核心结论 ----------
print("\n" + "=" * 56)
print("【核心结论】")
print(f"1. setosa 品种最容易区分(花萼特征即可分离)")
print(f"2. 花瓣特征 (长/宽) 区分度最高 ({sep_ratios['petal length (cm)']:.1f} 和 {sep_ratios['petal width (cm)']:.1f}), 是分类关键")
print(f"3. 花瓣长宽高度相关 (r={corr_pairs[0][2]:.2f}), 信息有重叠")
print(f"4. 花萼宽与其他特征弱相关/负相关, 是独立信息")
print("5. 用花瓣两个特征即可构建简单分类规则(线性可分)")
print("=" * 56)
print("IRIS_ANALYSIS_OK")
