#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iris + Kaggle 数据集下载 流程测试 — 简单 sklearn 训练
数据来源: notebook 第一 cell 用 kagglehub 下载的 uciml/iris (路径写在 /kaggle/working/data_path.txt)
"""
import sys, json, time
import pandas as pd
import numpy as np

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# 尝试打印 GPU 环境信息 (Kaggle 预装 torch)
try:
    import torch
    log(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
except Exception as e:
    log(f'torch 不可用: {e}')

# 读数据集路径 (notebook 第一 cell 下载后写入)
with open('/kaggle/working/data_path.txt') as f:
    data_path = f.read().strip()
log(f'data path: {data_path}')

df = pd.read_csv(f'{data_path}/Iris.csv')
log(f'iris 数据: {df.shape[0]} 行 × {df.shape[1]} 列, 列: {list(df.columns)}')

# 特征/标签 (Kaggle 版列名带 Id/Cm)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].values
log(f'类别: {sorted(set(y))}')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.time()
clf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))
secs = round(time.time() - t0, 2)
log(f'RF 训练完成 | 测试准确率 {acc:.4f} | {secs}s')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
imp = pd.Series(clf.feature_importances_, index=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
imp.sort_values().plot(kind='barh')
plt.title('Feature Importance (Random Forest on Kaggle iris)')
plt.tight_layout()
plt.savefig('/kaggle/working/iris_feature_importance.png', dpi=100)

with open('/kaggle/working/iris_result.json', 'w') as f:
    json.dump({'accuracy': acc, 'seconds': secs, 'data_source': 'kagglehub uciml/iris'}, f)
log(f'TRAIN_DONE acc={acc:.4f} seconds={secs}')
