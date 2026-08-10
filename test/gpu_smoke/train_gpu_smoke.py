#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DataScience-Kaggle skill v1.2.0 冒烟测试 — 简单 PyTorch GPU 训练
MLP + FashionMNIST, 2 epochs, 目标 1 分钟内跑完, 验证: GPU 可用/数据下载/训练/产物落盘"""
import sys, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)  # 铁律4: 进度走 stderr, log 页可见

log(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    log(f'GPU: {torch.cuda.get_device_name(0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据 (Kaggle 容器需联网下载, 提交时 --internet)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
log('downloading FashionMNIST...')
trainset = datasets.FashionMNIST(root='/kaggle/working/data', train=True, download=True, transform=tf)
loader = DataLoader(trainset, batch_size=64, shuffle=True)
log(f'train samples: {len(trainset)}')

# 简单 MLP: 784-128-10
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

EPOCHS = 2
losses = []
t0 = time.time()
for ep in range(EPOCHS):
    total, correct, running = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        running += loss.item()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        if i % 100 == 0:
            losses.append(loss.item())
            log(f'epoch {ep+1}/{EPOCHS} step {i} loss {loss.item():.4f} acc {correct/total:.3f}')
    log(f'epoch {ep+1} done | acc {correct/total:.3f} | {time.time()-t0:.1f}s')

# 产物落盘 (铁律12: 必须 savefig, 不能只 plt.show)
log('saving artifacts...')
torch.save(model.state_dict(), '/kaggle/working/model.pth')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title('training loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('/kaggle/working/loss_curve.png', dpi=100)
with open('/kaggle/working/result.json', 'w') as f:
    json.dump({'final_acc': correct / total, 'losses': losses, 'seconds': round(time.time() - t0, 1)}, f)
log(f'TRAIN_DONE final_acc={correct/total:.3f} seconds={time.time()-t0:.1f}')
