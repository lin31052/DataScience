#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heart Failure Prediction 分析 — 缺失值(条形图) → 核心统计 → 可视化 → 分类
数据: fedesoriano/heart-failure-prediction (918 行, 12 特征, 混合类型)
环境: Kaggle CPU (不烧 GPU 配额)
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

sns.set_theme(style="whitegrid", palette="muted")
def setup_chinese_font():
    import subprocess, glob as g
    found = []
    for pat in ["/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/*CJK*.ttc",
                "/usr/share/fonts/opentype/noto/*.ttc", "/usr/share/fonts/**/wqy*.ttc"]:
        found += g.glob(pat, recursive=True)
    if not found:
        log('中文字体缺失, apt-get install fonts-noto-cjk ...')
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        r = subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                           capture_output=True, text=True, timeout=300)
        log('apt rc=' + str(r.returncode))
        found = g.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + \
                g.glob("/usr/share/fonts/opentype/noto/*.ttc", recursive=True)
    for fp in sorted(set(found)):
        try: fm.fontManager.addfont(fp)
        except Exception: pass
    return sorted({f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ("CJK", "WenQuanYi"))})

zh = setup_chinese_font()
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    log('中文字体: ' + str(zh))

OUT = "/kaggle/working"
report = {}
df = pd.read_csv(glob.glob('/kaggle/working/kdata/heart/*.csv')[0])
log(f'heart: {df.shape[0]} 行 × {df.shape[1]} 列 | columns: {list(df.columns)}')
report['shape'] = [int(df.shape[0]), int(df.shape[1])]

# ---------- 1. 缺失值 (全局约定: 条形图) ----------
log("=" * 50); log("1. 缺失值分析")
miss_all = df.isnull().sum()
miss_info = {c: {'缺失数': int(miss_all[c]), '缺失率%': round(miss_all[c]/len(df)*100, 2)}
             for c in df.columns if miss_all[c] > 0}
for c, v in miss_info.items(): log(f"  {c}: {v}")
report['missing'] = miss_info
fig, ax = plt.subplots(figsize=(10, 5))
vals = [int(miss_all[c]) for c in df.columns]
bars = ax.bar(df.columns, vals, color=["#e74c3c" if v > 0 else "#95a5a6" for v in vals], edgecolor="white")
ax.set_title(f"缺失值数量分布（共 {int(miss_all.sum())} 个，总样本 {len(df)} 行）", fontsize=13)
ax.set_xlabel("列名"); ax.set_ylabel("缺失数量（条）")
for b, v in zip(bars, vals):
    if v > 0:
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v} 条\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
ax.set_ylim(0, max(vals)*1.4 if max(vals) > 0 else 1)
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT}/heart_01_missing_bar.png", dpi=110); plt.close()

# ---------- 2. 核心统计 ----------
log("=" * 50); log("2. 核心统计")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
log(f"数值列: {num_cols}"); log(f"类别列: {cat_cols}")
report['cat_cols'] = cat_cols
target = 'HeartDisease'
vc = df[target].value_counts()
log(f"目标分布: 无心脏病={vc.get(0,0)} ({vc.get(0,0)/len(df):.1%}) / 有心脏病={vc.get(1,0)} ({vc.get(1,0)/len(df):.1%})")
report['target_dist'] = vc.to_dict()
for c in cat_cols:
    log(f"  {c}: " + ", ".join(f"{k}={v}" for k, v in df[c].value_counts().items()))
report['describe'] = df[num_cols].describe().round(2).to_dict()

# ---------- 3. 可视化 ----------
# 3.1 目标分布
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ["#4C72B0", "#e74c3c"]
bars = ax.bar(["无心脏病", "有心脏病"], [vc.get(0,0), vc.get(1,0)], color=colors, edgecolor="white")
for b, v in zip(bars, [vc.get(0,0), vc.get(1,0)]):
    ax.text(b.get_x()+b.get_width()/2, v+15, f"{v} 例\n({v/len(df):.1%})", ha="center", fontsize=11)
ax.set_title("心脏病目标分布（较均衡）", fontsize=14)
ax.set_ylabel("人数")
plt.tight_layout(); plt.savefig(f"{OUT}/heart_02_target.png", dpi=110); plt.close()

# 3.2 关键类别特征 × 目标
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, c in zip(axes, ["ChestPainType", "ST_Slope"]):
    ct = pd.crosstab(df[c], df[target], normalize='index') * 100
    ct.columns = ["无心脏病", "有心脏病"]
    ct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white')
    ax.set_title(f"{c} × 心脏病比例", fontsize=13)
    ax.set_xlabel(c); ax.set_ylabel("比例 %"); ax.legend(title='', fontsize=9)
    for i in range(len(ct)):
        y = 0
        for j in range(2):
            v = ct.iloc[i, j]
            if v > 3: ax.text(i, y + v/2, f"{v:.0f}%", ha='center', fontsize=9)
            y += v
plt.tight_layout(); plt.savefig(f"{OUT}/heart_03_cat_target.png", dpi=110); plt.close()

# 3.3 数值特征 KDE × 目标
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, c in zip(axes, ["Age", "MaxHR", "Oldpeak"]):
    for v, lab, col in [(0, "无心脏病", "#4C72B0"), (1, "有心脏病", "#e74c3c")]:
        sns.kdeplot(df.loc[df[target] == v, c].dropna(), label=lab, ax=ax, fill=True, alpha=0.3, color=col)
    ax.set_title(f"{c} 分布", fontsize=12)
plt.suptitle("数值特征 × 心脏病（MaxHR/Oldpeak 区分度明显）", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/heart_04_num_kde.png", dpi=110); plt.close()

# 3.4 相关性
fig, ax = plt.subplots(figsize=(8, 6.5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot_kws={"fontsize": 9})
ax.set_title("数值特征相关性热力图", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/heart_05_corr.png", dpi=110); plt.close()
report['corr'] = corr.round(3).to_dict()

# ---------- 4. 分类 ----------
log("=" * 50); log("4. 分类 (RF vs LR)")
dfc = df.copy()
for c in cat_cols:
    dfc[c] = LabelEncoder().fit_transform(dfc[c].astype(str))
X = dfc.drop(columns=[target]); y = dfc[target]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
log(f"训练 {len(X_tr)} / 测试 {len(X_te)}")
for name, model in [("随机森林", RandomForestClassifier(n_estimators=100, random_state=42)),
                    ("逻辑回归", LogisticRegression(max_iter=500, random_state=42))]:
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    log(f"  {name}: acc={acc:.4f}")
    report[f'acc_{name}'] = float(acc)
    if name == "随机森林":
        cm = confusion_matrix(y_te, model.predict(X_te))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["无", "有"], yticklabels=["无", "有"])
        ax.set_title(f"随机森林混淆矩阵 (acc={acc:.4f})", fontsize=13)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        plt.tight_layout(); plt.savefig(f"{OUT}/heart_06_confusion.png", dpi=110); plt.close()

with open(f"{OUT}/heart_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=str)
log('HEART_DONE')
