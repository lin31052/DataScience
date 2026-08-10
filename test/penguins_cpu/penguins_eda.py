#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""企鹅数据集 CPU 数据分析 — 缺失值 → 核心统计 → 可视化 → 快速分类
数据: parulpandey/palmer-archipelago-antarctica-penguin-data (344 行, 3 物种, 有真实缺失值)
环境: Kaggle CPU (不烧 GPU 配额), pandas/sklearn/matplotlib/seaborn
"""
import sys, json, glob, os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ---------- 中文字体 (Kaggle 容器默认无中文字体 → 先 apt 安装再注册) ----------
sns.set_theme(style="whitegrid", palette="muted")
def setup_chinese_font():
    """Kaggle 容器版中文字体: 存在则直接用, 不存在则 apt 安装 fonts-noto-cjk"""
    import subprocess, glob
    found = []
    for pat in ["/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/*CJK*.ttc",
                "/usr/share/fonts/opentype/noto/*.ttc", "/usr/share/fonts/**/wqy*.ttc"]:
        found += glob.glob(pat, recursive=True)
    if not found:
        log('中文字体缺失, 执行 apt-get install fonts-noto-cjk ...')
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        r = subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                           capture_output=True, text=True, timeout=300)
        log('apt result: rc=' + str(r.returncode))
        found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + \
                glob.glob("/usr/share/fonts/opentype/noto/*.ttc", recursive=True)
    for fp in sorted(set(found)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    zh = sorted({f.name for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ("CJK", "WenQuanYi"))})
    return zh

zh = setup_chinese_font()
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    log(f'中文字体: {zh}')
else:
    log('WARNING: 未找到中文字体, 图内中文可能显示为方块')

OUT = "/kaggle/working"
report = {}

# ---------- 数据 ----------
with open('/kaggle/working/data_path.txt') as f:
    data_path = f.read().strip()
log(f'data path: {data_path}')
log('files: ' + str(os.listdir(data_path)))

cands = [f for f in glob.glob(f'{data_path}/*.csv') if 'size' in os.path.basename(f).lower()]
if not cands:
    cands = glob.glob(f'{data_path}/*.csv')
df = pd.read_csv(cands[0])
df.columns = [c.strip().lower() for c in df.columns]
log(f'penguins: {df.shape[0]} 行 × {df.shape[1]} 列')
log('columns: ' + str(list(df.columns)))
report['shape'] = [int(df.shape[0]), int(df.shape[1])]

# ---------- 1. 缺失值分析 ----------
log("=" * 50)
log("1. 缺失值分析")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
miss_info = {c: {'缺失数': int(missing[c]), '缺失率%': float(missing_pct[c])}
             for c in df.columns if missing[c] > 0}
for c, info in miss_info.items():
    log(f"  {c}: 缺失 {info['缺失数']} 条 ({info['缺失率%']}%)")
report['missing'] = miss_info

fig, ax = plt.subplots(figsize=(9, 5))
miss_all = df.isnull().sum()
cols_all = df.columns.tolist()
vals_all = [int(miss_all[c]) for c in cols_all]
pcts_all = [round(miss_all[c] / len(df) * 100, 1) for c in cols_all]
bars = ax.bar(cols_all, vals_all,
              color=["#e74c3c" if v > 0 else "#95a5a6" for v in vals_all],
              edgecolor="white")
ax.set_title(f"缺失值数量分布（共 {int(miss_all.sum())} 个，总样本 {len(df)} 行）", fontsize=14)
ax.set_xlabel("列名"); ax.set_ylabel("缺失数量（条）")
for b, v, p in zip(bars, vals_all, pcts_all):
    if v > 0:
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v} 条\n({p}%)",
                ha="center", fontsize=10)
ax.set_ylim(0, max(vals_all) * 1.35 if max(vals_all) > 0 else 1)
plt.tight_layout()
plt.savefig(f"{OUT}/01_missing_bar.png", dpi=110)
plt.close()

# ---------- 2. 核心统计 ----------
log("=" * 50)
log("2. 核心统计")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
log(f"数值列: {num_cols}")
log(f"类别列: {cat_cols}")
for c in cat_cols:
    vc = df[c].value_counts(dropna=False)
    log(f"  {c} 分布: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
report['cat_dist'] = {c: df[c].value_counts(dropna=False).to_dict() for c in cat_cols}
report['describe'] = df[num_cols].describe().round(2).to_dict()

# ---------- 3. 可视化 ----------
# 3.1 类别分布: 物种 × 岛屿 / 物种 × 性别
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ct1 = pd.crosstab(df['species'], df['island'])
ct1.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='white')
axes[0].set_title("物种 × 岛屿分布", fontsize=13)
axes[0].set_xlabel("物种"); axes[0].set_ylabel("数量"); axes[0].legend(title='岛屿')
ct2 = pd.crosstab(df['species'], df['sex'])
ct2.plot(kind='bar', ax=axes[1], colormap='Set1', edgecolor='white')
axes[1].set_title("物种 × 性别分布", fontsize=13)
axes[1].set_xlabel("物种"); axes[1].set_ylabel("数量"); axes[1].legend(title='性别')
plt.tight_layout()
plt.savefig(f"{OUT}/02_category_counts.png", dpi=110)
plt.close()

# 3.2 关键特征 KDE 按物种
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for i, c in enumerate(num_cols):
    ax = axes[i // 2, i % 2]
    for sp in df['species'].dropna().unique():
        sns.kdeplot(df.loc[df['species'] == sp, c].dropna(), label=sp, ax=ax, fill=True, alpha=0.25)
    ax.set_title(f"{c} 分布（按物种）", fontsize=12)
    ax.set_xlabel(c)
plt.suptitle("数值特征分布（物种间区分明显）", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/03_feature_kde.png", dpi=110)
plt.close()

# 3.3 箱线图
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for i, c in enumerate(num_cols):
    ax = axes[i // 2, i % 2]
    sns.boxplot(data=df, x='species', y=c, ax=ax, palette='muted')
    ax.set_title(f"{c} 箱线图", fontsize=12)
    ax.set_xlabel("物种")
plt.suptitle("物种间特征差异（特征可区分性）", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/04_boxplot_features.png", dpi=110)
plt.close()

# 3.4 相关性热力图
fig, ax = plt.subplots(figsize=(8, 6.5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"fontsize": 10})
ax.set_title("数值特征相关性热力图", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/05_corr_heatmap.png", dpi=110)
plt.close()
report['corr'] = corr.round(3).to_dict()

# 3.5 pairplot (344 行, 秒级)
g = sns.pairplot(df.dropna(subset=num_cols + ['species']), hue='species',
                 vars=num_cols, diag_kind='kde', height=2.2, palette='deep')
g.fig.suptitle("特征配对关系（按物种着色）", y=1.02, fontsize=15)
g.savefig(f"{OUT}/06_pairplot.png", dpi=110)
plt.close()

# ---------- 4. 快速分类 (CPU 秒级) ----------
log("=" * 50)
log("4. 快速分类 (随机森林 vs 逻辑回归)")
dfc = df.dropna(subset=num_cols + ['species']).copy()
X = dfc[num_cols].fillna(dfc[num_cols].median())
y = dfc['species']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
log(f"训练 {len(X_tr)} / 测试 {len(X_te)}")
for name, model in [("随机森林", RandomForestClassifier(n_estimators=100, random_state=42)),
                    ("逻辑回归", LogisticRegression(max_iter=500, random_state=42))]:
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    log(f"  {name}: 准确率 {acc:.4f}")
    report[f'acc_{name}'] = float(acc)
    if name == "随机森林":
        cm = confusion_matrix(y_te, model.predict(X_te))
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=model.classes_, yticklabels=model.classes_)
        ax.set_title(f"随机森林混淆矩阵 (acc={acc:.4f})", fontsize=13)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        plt.tight_layout()
        plt.savefig(f"{OUT}/07_confusion.png", dpi=110)
        plt.close()

with open(f"{OUT}/penguins_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=str)
log("ARTIFACTS: " + str(sorted(os.listdir(OUT))))
log('ALL_DONE')
