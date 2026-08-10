#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGBoost 安装验证 + titanic 简单训练/可视化 (CPU)
本地沙箱(aarch64/musl)装不了 xgboost, 验证 Kaggle CPU 环境可装可跑"""
import sys, json, os
import pandas as pd
import numpy as np

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ---------- XGBoost 导入验证 ----------
try:
    import xgboost as xgb
    log(f'XGBoost import OK, version: {xgb.__version__}')
except Exception as e:
    log(f'XGBoost import FAILED: {e}')
    sys.exit(1)

# ---------- 数据 ----------
with open('/kaggle/working/data_path.txt') as f:
    data_path = f.read().strip()
cands = [f for f in os.listdir(data_path) if f.lower().endswith('.csv') and 'train' in f.lower()]
if not cands:
    cands = [f for f in os.listdir(data_path) if f.lower().endswith('.csv')]
df = pd.read_csv(os.path.join(data_path, cands[0]))
log(f'titanic: {df.shape[0]} 行 × {df.shape[1]} 列')

# ---------- 简单预处理 ----------
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'], errors='ignore')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['HasCabin'] = df['Cabin'].notna().astype(int)
df = df.drop(columns=['Cabin'], errors='ignore')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df = df.dropna()

X = df.drop(columns=['Survived'])
y = df['Survived']
log(f'特征: {list(X.columns)}, 样本: {len(X)}')

# ---------- XGBoost 训练 (CPU) ----------
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
clf.fit(X_tr, y_tr)
acc = clf.score(X_te, y_te)
log(f'XGBoost 训练完成, 测试准确率: {acc:.4f}')

# ---------- 可视化: 特征重要性 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
imp.plot(kind='barh', color='#4C72B0')
plt.title(f'Titanic Feature Importance (XGBoost, acc={acc:.3f})')
plt.xlabel('importance')
plt.tight_layout()
plt.savefig('/kaggle/working/xgb_titanic.png', dpi=100)

with open('/kaggle/working/xgb_result.json', 'w') as f:
    json.dump({'xgb_version': xgb.__version__, 'acc': float(acc), 'features': list(X.columns)}, f)
log(f'XGB_DONE acc={acc:.4f} 产物: xgb_titanic.png + xgb_result.json')
