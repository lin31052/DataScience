#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iris 神经网络训练 (T4 GPU) — Kaggle 数据集下载 → torch MLP 训练
数据: notebook cell1 用 kaggle CLI 下载的 uciml/iris, 路径在 /kaggle/working/data_path.txt
模型: MLP 4→16→3, iris 150 样本, GPU 秒级训练
"""
import sys, json, time
import pandas as pd
import numpy as np

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ---------- 数据 ----------
with open('/kaggle/working/data_path.txt') as f:
    data_path = f.read().strip()
log(f'data path: {data_path}')

df = pd.read_csv(f'{data_path}/Iris.csv')
log(f'iris: {df.shape[0]} 行 × {df.shape[1]} 列')

# 特征/标签 → 数值
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values.astype(np.float32)
species = sorted(df['Species'].unique())
y = np.array([species.index(s) for s in df['Species']], dtype=np.int64)
log(f'类别映射: {dict(enumerate(species))}')

# ---------- torch + GPU ----------
import torch
import torch.nn as nn
log(f'torch {torch.__version__}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))

# ---------- 简单 MLP ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 3),
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# 70/30 划分
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_tr, y_tr = torch.tensor(X_tr).to(device), torch.tensor(y_tr).to(device)
X_te, y_te = torch.tensor(X_te).to(device), torch.tensor(y_te).to(device)

EPOCHS = 100
losses = []
t0 = time.time()
for ep in range(EPOCHS):
    opt.zero_grad()
    out = model(X_tr)
    loss = crit(out, y_tr)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if (ep + 1) % 20 == 0:
        log(f'epoch {ep+1}/{EPOCHS} loss {loss.item():.4f}')

with torch.no_grad():
    acc = (model(X_te).argmax(1) == y_te).float().mean().item()
secs = round(time.time() - t0, 2)
log(f'TRAIN_DONE test_acc={acc:.4f} seconds={secs}')

# ---------- 产物 ----------
torch.save(model.state_dict(), '/kaggle/working/iris_nn_model.pth')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title(f'MLP training loss (iris, acc={acc:.3f})')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('/kaggle/working/iris_nn_loss.png', dpi=100)
with open('/kaggle/working/iris_nn_result.json', 'w') as f:
    json.dump({'test_acc': acc, 'seconds': secs, 'device': str(device),
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
               'data_source': 'kaggle CLI download uciml/iris'}, f)
log('ARTIFACTS_SAVED')
