#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车牌字符识别训练 — 简单 CNN (GPU T4) + 结果可视化
数据: aladdinss/license-plate-digits-classification-dataset (34 类, 35500 张, 75x100)
模型: 3 层卷积 CNN, 15 epochs, batch 64
"""
import sys, json, os, time
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

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
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                       capture_output=True, text=True, timeout=300)
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
DATA = "/kaggle/working/kdata/plate_char/CNN letter Dataset"
report = {}

# ---------- 1. 数据加载 (ImageFolder, 目录名=标签) ----------
log(f"CUDA: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
tf = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
ds = datasets.ImageFolder(DATA, transform=tf)
classes = ds.classes
log(f"加载完成: {len(ds)} 张, {len(classes)} 类: {''.join(classes)}")
report['n_total'] = len(ds); report['classes'] = classes

# ---------- 1.5 按前缀分组划分 (group split, 增强数据集必须同原始样本同侧) ----------
import re as _re
prefixes = []
for path, _ in ds.samples:
    base = os.path.basename(path)
    m = _re.match(r'(.+?)_\d+\.jpg$', base)
    prefixes.append(m.group(1) if m else base)
uniq_pref = np.array(sorted(set(prefixes)))
rng = np.random.RandomState(42)
n_val_pref = int(len(uniq_pref) * 0.2)
val_pref = set(rng.choice(uniq_pref, size=n_val_pref, replace=False).tolist())
tr_idx = [i for i, p in enumerate(prefixes) if p not in val_pref]
va_idx = [i for i, p in enumerate(prefixes) if p in val_pref]
n_twin = sum(1 for p in prefixes if p in val_pref)
log(f"分组划分: 唯一前缀 {len(uniq_pref)} (验证前缀 {len(val_pref)}) | 训练 {len(tr_idx)} / 验证 {len(va_idx)} | 验证孪生对 0 (已同侧)")
train_ds = torch.utils.data.Subset(ds, tr_idx)
val_ds = torch.utils.data.Subset(ds, va_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

# ---------- 2. CNN 模型 ----------
class PlateCNN(nn.Module):
    def __init__(self, n_classes=34):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 12, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = PlateCNN(len(classes)).cuda()
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)
log(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ---------- 3. 训练 ----------
EPOCHS = 15
hist = {'loss': [], 'acc': []}
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    tot_loss, corr, n = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward(); opt.step()
        tot_loss += loss.item() * len(xb)
        corr += (out.argmax(1) == yb).sum().item()
        n += len(xb)
    model.eval()
    v_corr, v_n = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.cuda(), yb.cuda()
            v_corr += (model(xb).argmax(1) == yb).sum().item()
            v_n += len(xb)
    tr_loss = tot_loss / n; tr_acc = corr / n; va_acc = v_corr / v_n
    hist['loss'].append(round(tr_loss, 4)); hist['acc'].append(round(va_acc, 4))
    log(f"Epoch {ep}/{EPOCHS} | loss={tr_loss:.4f} | train_acc={tr_acc:.4f} | val_acc={va_acc:.4f} | {time.time()-t0:.0f}s")
report['history'] = hist
report['seconds'] = round(time.time() - t0, 1)
log(f"TRAIN_DONE 用时 {report['seconds']}s")

# ---------- 4. 验证集最终评估 ----------
model.eval()
all_pred, all_y = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.cuda()
        all_pred += model(xb).argmax(1).cpu().tolist()
        all_y += yb.tolist()
final_acc = (np.array(all_pred) == np.array(all_y)).mean()
report['final_acc'] = round(float(final_acc), 4)
log(f"最终验证准确率: {final_acc:.4f}")

# ---------- 5. 可视化 ----------
# 5.1 训练曲线 (中文)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].plot(hist['loss'], marker='o', color="#4C72B0")
axes[0].set_title("训练损失曲线", fontsize=13); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[1].plot(hist['acc'], marker='o', color="#e74c3c")
axes[1].set_title(f"验证准确率曲线 (最终 {final_acc:.4f})", fontsize=13)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("acc")
plt.tight_layout(); plt.savefig(f"{OUT}/char_group_03_metrics_zh.png", dpi=110); plt.close()

# 5.2 混淆矩阵 (34x34)
cm = confusion_matrix(all_y, all_pred)
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True,
            xticklabels=classes, yticklabels=classes, annot_kws={"fontsize": 5})
ax.set_title(f"34 类混淆矩阵（acc={final_acc:.4f}）", fontsize=14)
ax.set_xlabel("预测"); ax.set_ylabel("真实")
plt.tight_layout(); plt.savefig(f"{OUT}/char_group_04_confusion.png", dpi=110); plt.close()

# 5.3 错误样本分析 (top 易混对)
cmn = cm.copy(); np.fill_diagonal(cmn, 0)
pairs = [(cmn[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j]
pairs.sort(reverse=True)
log("Top 易混对: " + ", ".join(f"{a}→{b}({v})" for v, a, b in pairs[:8]))
report['top_confusions'] = [{'true': a, 'pred': b, 'count': int(v)} for v, a, b in pairs[:8]]

# 5.4 预测示例 (验证集 12 张)
wrong_idx = [i for i, (p, y) in enumerate(zip(all_pred, all_y)) if p != y]
sample_idx = wrong_idx[:6] + [i for i in range(len(all_y)) if all_pred[i] == all_y[i]][:6]
fig, axes = plt.subplots(3, 4, figsize=(13, 9))
for ax, i in zip(axes.ravel(), sample_idx):
    img, label = val_ds[i]
    pred = all_pred[i]
    ok = pred == label
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"真:{classes[label]} 预:{classes[pred]} {'✓' if ok else '✗'}", fontsize=10,
                 color="#2e7d32" if ok else "#c62828")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("验证集预测示例（绿✓=对 红✗=错）", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/char_group_05_predictions.png", dpi=110); plt.close()

with open(f"{OUT}/char_group_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
log('CHAR_TRAIN_DONE')
