#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stroke Prediction 分析 — 缺失值条形图 + 类别不平衡 + 可视化 + 分类
数据: fedesoriano/stroke-prediction-dataset (5110 行, 12 特征)
教学点: ①bmi 缺失 201 条 (3.9%) ②类别不平衡 (中风仅 4.8%) → class_weight 对比
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
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
df = pd.read_csv(glob.glob('/kaggle/working/kdata/stroke/*.csv')[0])
log(f'stroke: {df.shape[0]} 行 × {df.shape[1]} 列 | columns: {list(df.columns)}')
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
        ax.text(b.get_x()+b.get_width()/2, v+3, f"{v} 条\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
ax.set_ylim(0, max(vals)*1.4 if max(vals) > 0 else 1)
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT}/stroke_01_missing_bar.png", dpi=110); plt.close()

# ---------- 2. 核心统计 + 类别不平衡 ----------
log("=" * 50); log("2. 核心统计 + 类别不平衡")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
log(f"数值列: {num_cols}"); log(f"类别列: {cat_cols}")
target = 'stroke'
vc = df[target].value_counts()
n_pos = vc.get(1, 0)
log(f"目标分布: 无中风={vc.get(0,0)} ({vc.get(0,0)/len(df):.2%}) / 中风={n_pos} ({n_pos/len(df):.2%})")
log(f"⚠️ 类别不平衡: 中风仅 {n_pos/len(df):.2%}, 全猜'无'也有 {vc.get(0,0)/len(df):.2%} 准确率")
report['target_dist'] = vc.to_dict()
report['imbalance'] = {'pos_rate': round(n_pos/len(df), 4)}

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["无中风", "中风"], [vc.get(0,0), n_pos], color=["#4C72B0", "#e74c3c"], edgecolor="white")
for b, v in zip(bars, [vc.get(0,0), n_pos]):
    ax.text(b.get_x()+b.get_width()/2, v+60, f"{v} 例\n({v/len(df):.2%})", ha="center", fontsize=11)
ax.set_title("中风目标分布（严重不平衡，中风仅 4.8%）", fontsize=14)
ax.set_ylabel("人数")
plt.tight_layout(); plt.savefig(f"{OUT}/stroke_02_target.png", dpi=110); plt.close()

# 缺失行分析: bmi 缺失的样本年龄分布
bmi_miss = df['bmi'].isnull()
log(f"bmi 缺失样本的平均年龄: {df.loc[bmi_miss,'age'].mean():.1f} vs 完整样本: {df.loc[~bmi_miss,'age'].mean():.1f}")
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.kdeplot(df.loc[~bmi_miss, 'age'], label='bmi 完整', fill=True, alpha=0.3, color="#4C72B0", ax=ax)
sns.kdeplot(df.loc[bmi_miss, 'age'], label='bmi 缺失', fill=True, alpha=0.4, color="#e74c3c", ax=ax)
ax.set_title("bmi 缺失 vs 完整的年龄分布（缺失偏高龄）", fontsize=13)
ax.set_xlabel("年龄"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/stroke_03_bmi_missing.png", dpi=110); plt.close()

# ---------- 3. 特征可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, c in zip(axes, ["work_type", "smoking_status"]):
    ct = pd.crosstab(df[c], df[target], normalize='index') * 100
    ct.columns = ["无中风", "中风"]
    ct.plot(kind='bar', stacked=True, ax=ax, color=["#4C72B0", "#e74c3c"], edgecolor='white')
    ax.set_title(f"{c} × 中风比例", fontsize=13)
    ax.set_xlabel(c); ax.set_ylabel("比例 %"); ax.legend(title='', fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/stroke_04_cat_target.png", dpi=110); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, c in zip(axes, ["age", "avg_glucose_level", "bmi"]):
    for v, lab, col in [(0, "无中风", "#4C72B0"), (1, "中风", "#e74c3c")]:
        sns.kdeplot(df.loc[df[target] == v, c].dropna(), label=lab, ax=ax, fill=True, alpha=0.3, color=col)
    ax.set_title(f"{c} 分布", fontsize=12)
plt.suptitle("数值特征 × 中风（高龄段中风占比明显升高）", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/stroke_05_num_kde.png", dpi=110); plt.close()

# ---------- 4. 分类: 不平衡处理对比 ----------
log("=" * 50); log("4. 分类: 默认 vs class_weight=balanced")
dfc = df.drop(columns=['id']).copy()
for c in dfc.select_dtypes(include=["object"]).columns:
    dfc[c] = LabelEncoder().fit_transform(dfc[c].astype(str))
dfc['bmi'] = dfc['bmi'].fillna(dfc['bmi'].median())
X = dfc.drop(columns=[target]); y = dfc[target]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
log(f"训练 {len(X_tr)} / 测试 {len(X_te)} (测试集中风 {int(y_te.sum())} 例)")
results = {}
for name, cw in [("默认(不平衡)", None), ("balanced(加权)", 'balanced')]:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=cw)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    acc = accuracy_score(y_te, pred); rec = recall_score(y_te, pred); f1 = f1_score(y_te, pred)
    log(f"  RF {name}: acc={acc:.4f} | 中风召回率={rec:.4f} | F1={f1:.4f}")
    results[name] = {'acc': round(float(acc), 4), 'recall': round(float(rec), 4), 'f1': round(float(f1), 4)}
    if 'balanced' in name:
        cm = confusion_matrix(y_te, pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["无", "有"], yticklabels=["无", "有"])
        ax.set_title(f"balanced RF 混淆矩阵 (acc={acc:.4f}, 召回={rec:.4f})", fontsize=12)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        plt.tight_layout(); plt.savefig(f"{OUT}/stroke_06_confusion.png", dpi=110); plt.close()
report['models'] = results

with open(f"{OUT}/stroke_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=str)
log('STROKE_DONE')
