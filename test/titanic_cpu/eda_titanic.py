#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""titanic CPU 数据分析 (无 GPU) — 缺失值处理 + 常见 EDA
数据: notebook cell1 用 kaggle CLI 下载的 heptapod/titanic, 路径在 /kaggle/working/data_path.txt
全程 CPU (pandas/sklearn), 验证 skill 在 CPU 模式下也能跑
"""
import sys, json, glob, os
import pandas as pd
import numpy as np

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ---------- 数据 ----------
with open('/kaggle/working/data_path.txt') as f:
    data_path = f.read().strip()
log(f'data path: {data_path}')
log('files: ' + str(os.listdir(data_path)))

# 自动找 train 文件 (兼容 train.csv / Train.csv 等)
cands = [f for f in glob.glob(f'{data_path}/*.csv') if 'train' in os.path.basename(f).lower()]
if not cands:
    cands = glob.glob(f'{data_path}/*.csv')
log(f'using: {os.path.basename(cands[0])}')
df = pd.read_csv(cands[0])
log(f'titanic: {df.shape[0]} 行 × {df.shape[1]} 列')

report = {}

# ---------- 1. 缺失值统计 ----------
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
miss_info = {c: {'缺失数': int(missing[c]), '缺失率%': float(missing_pct[c])}
             for c in df.columns if missing[c] > 0}
log(f'缺失值列: {list(miss_info.keys())}')
report['missing_before'] = miss_info

# ---------- 2. 缺失值处理 ----------
# Age: 中位数填充 (按性别分组的更合理, 这里用整体中位数演示)
if 'Age' in df.columns and df['Age'].isnull().any():
    df['Age'] = df['Age'].fillna(df['Age'].median())
# Embarked: 众数填充
if 'Embarked' in df.columns and df['Embarked'].isnull().any():
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Cabin: 有/无标记 (缺失即无)
if 'Cabin' in df.columns:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df.drop(columns=['Cabin'], inplace=True)
# 剩余数值缺失: 0 填充; 类别缺失: 'Unknown'
for c in df.columns:
    if df[c].isnull().any():
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna('Unknown')

remaining = int(df.isnull().sum().sum())
log(f'缺失值处理后剩余缺失: {remaining}')
report['missing_after'] = remaining

# ---------- 3. 常见数据分析 ----------
report['total_passengers'] = int(len(df))
if 'Survived' in df.columns:
    surv = df['Survived'].mean()
    report['survival_rate'] = round(float(surv), 4)
    log(f'总体生存率: {surv:.2%}')

    # 按性别
    g = df.groupby('Sex')['Survived'].mean().to_dict()
    report['by_sex'] = {k: round(float(v), 4) for k, v in g.items()}
    log(f'性别生存率: {g}')

    # 按舱位
    if 'Pclass' in df.columns:
        g = df.groupby('Pclass')['Survived'].mean().to_dict()
        report['by_pclass'] = {str(k): round(float(v), 4) for k, v in g.items()}
        log(f'舱位生存率: {g}')

    # 按年龄组
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 40, 60, 100], labels=['儿童', '少年', '青年', '中年', '老年'])
        g = df.groupby('AgeGroup', observed=True)['Survived'].mean().to_dict()
        report['by_age_group'] = {k: round(float(v), 4) for k, v in g.items()}
        log(f'年龄组生存率: {g}')

# 数值特征统计
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ('Survived', 'PassengerId')][:6]
report['numeric_stats'] = df[num_cols].describe().loc[['mean', 'std', 'min', 'max']].round(2).to_dict()

# ---------- 4. 可视化 (CPU, savefig) ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
if 'Survived' in df.columns:
    df['Survived'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#d62728', '#2ca02c'])
    axes[0, 0].set_title('Survival Count')
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0, 1], color=['#1f77b4', '#ff7f0e'])
    axes[0, 1].set_title('Survival Rate by Sex')
if 'Age' in df.columns:
    df['Age'].hist(bins=20, ax=axes[1, 0], color='#9467bd')
    axes[1, 0].set_title('Age Distribution (filled)')
if 'Pclass' in df.columns:
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[1, 1], color='#8c564b')
    axes[1, 1].set_title('Survival Rate by Pclass')
plt.tight_layout()
plt.savefig('/kaggle/working/titanic_eda.png', dpi=100)

with open('/kaggle/working/titanic_eda_result.json', 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
log(f'EDA_DONE 产物: titanic_eda.png + titanic_eda_result.json')
log(f'SUMMARY: 总人数 {report.get("total_passengers")}, 生存率 {report.get("survival_rate")}, 处理后缺失 {remaining}')
