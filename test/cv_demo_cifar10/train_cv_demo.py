#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIFAR-10 计算机视觉 demo（T4 GPU）
数据分析/可视化 → 小 CNN 训练 3 epochs → 评估可视化
数据集: torchvision 自带 CIFAR-10（无需 Kaggle Dataset，首次自动下载 ~170MB）
产物: /kaggle/working/out/ 下 5 张中文图 + result.json + 模型权重
"""
import json
import os
import subprocess
import sys
import time
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    print("PROGRESS: " + msg, file=sys.stderr, flush=True)  # 铁律6: 进度走 stderr


# ---------- 中文字体（铁律17: Kaggle 容器默认没有中文字体）----------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402


def setup_chinese():
    found = []
    for pat in [
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/*CJK*.ttc",
        "/usr/share/fonts/opentype/noto/*.ttc",
        "/usr/share/fonts/**/wqy*.ttc",
    ]:
        found += glob.glob(pat, recursive=True)
    if not found:
        log("中文字体缺失，apt-get install fonts-noto-cjk ...")
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        r = subprocess.run(
            ["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
            capture_output=True, text=True, timeout=300,
        )
        log("apt result: rc=" + str(r.returncode))
        found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + glob.glob(
            "/usr/share/fonts/opentype/noto/*.ttc", recursive=True
        )
    for fp in sorted(set(found)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    return sorted({f.name for f in fm.fontManager.ttflist if "CJK" in f.name or "WenQuanYi" in f.name})


zh = setup_chinese()
log("中文字体: " + str(zh[:3]))
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

CLASS_ZH = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
N_CLASS = 10

# ---------- 设备 ----------
log(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

# ---------- 数据 ----------
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

log("加载 CIFAR-10（首次会下载）...")
t_load = time.time()
train_ds = datasets.CIFAR10("/kaggle/working/data", True, download=True, transform=tf_train)
test_ds = datasets.CIFAR10("/kaggle/working/data", False, download=True, transform=tf_test)
raw_ds = datasets.CIFAR10("/kaggle/working/data", False, download=True, transform=transforms.ToTensor())
log(f"train {len(train_ds)} / test {len(test_ds)} | {time.time()-t_load:.1f}s")

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# ---------- 可视化 1: 类别分布 ----------
counts = [0] * N_CLASS
for y in train_ds.targets:
    counts[y] += 1
log("类别分布: " + str(counts))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_ZH, counts, color="#4C78A8")
ax.bar_label(bars, fontsize=8)
ax.set_title(f"CIFAR-10 训练集类别分布（共 {sum(counts)} 张）")
ax.set_ylabel("样本数")
ax.set_ylim(0, max(counts) * 1.15)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/01_class_dist.png", dpi=120)
plt.close()

# ---------- 可视化 2: 样本网格 ----------
fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
for i, ax in enumerate(axes.flat):
    img, label = raw_ds[i]
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(CLASS_ZH[label], fontsize=9)
    ax.axis("off")
fig.suptitle("CIFAR-10 样本示例（每行同一类别）", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/02_samples.png", dpi=120)
plt.close()


# ---------- 模型 ----------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),      # 32 -> 16
            conv_block(32, 64),     # 16 -> 8
            conv_block(64, 128),    # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, N_CLASS))

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN().to(device)
n_param = sum(p.numel() for p in model.parameters())
log(f"模型参数量: {n_param/1e6:.2f}M")

EPOCHS, LR = 3, 2e-3
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(train_loader))
crit = nn.CrossEntropyLoss()

hist = {"train_loss": [], "train_acc": [], "test_acc": []}
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    running, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    tr_loss, tr_acc = running / total, correct / total

    model.eval()
    c_acc = c_tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            c_acc += (pred == y).sum().item()
            c_tot += y.size(0)
    te_acc = c_acc / c_tot

    hist["train_loss"].append(tr_loss)
    hist["train_acc"].append(tr_acc)
    hist["test_acc"].append(te_acc)
    log(f"epoch {ep+1}/{EPOCHS} loss {tr_loss:.4f} train_acc {tr_acc:.4f} test_acc {te_acc:.4f} | {time.time()-t0:.1f}s")

# ---------- 评估: 混淆矩阵 + 每类准确率 ----------
import numpy as np  # noqa: E402

cm = np.zeros((N_CLASS, N_CLASS), dtype=int)
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        for t, p in zip(y.tolist(), model(x).argmax(1).tolist()):
            cm[t][p] += 1
per_class = [cm[i][i] / max(cm[i].sum(), 1) for i in range(N_CLASS)]
final_acc = float(np.trace(cm) / cm.sum())
log(f"eval done: acc {final_acc:.4f}")

# 可视化 3: 训练曲线
fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
eps = list(range(1, EPOCHS + 1))
ax1.plot(eps, hist["train_loss"], "o-", color="#E45756", label="训练损失")
ax1.set_xlabel("epoch")
ax1.set_ylabel("损失", color="#E45756")
ax2 = ax1.twinx()
ax2.plot(eps, hist["train_acc"], "s--", color="#4C78A8", label="训练准确率")
ax2.plot(eps, hist["test_acc"], "^-", color="#54A24B", label="测试准确率")
ax2.set_ylabel("准确率")
ax2.set_ylim(0, 1)
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
ax1.set_title(f"训练过程（{gpu_name}，{n_param/1e6:.2f}M 参数）")
plt.tight_layout()
plt.savefig(f"{OUT}/03_train_curves.png", dpi=120)
plt.close()

# 可视化 4: 混淆矩阵
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(N_CLASS), CLASS_ZH, rotation=45)
ax.set_yticks(range(N_CLASS), CLASS_ZH)
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
ax.set_title(f"CIFAR-10 混淆矩阵（测试集 acc={final_acc:.3f}）")
for i in range(N_CLASS):
    for j in range(N_CLASS):
        if cm[i][j]:
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i][j] > cm.max() * 0.5 else "#333")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{OUT}/04_confusion.png", dpi=120)
plt.close()

# 可视化 5: 每类准确率
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_ZH, [a * 100 for a in per_class], color="#72B7B2")
ax.bar_label(bars, fmt="%.1f", fontsize=8)
ax.axhline(final_acc * 100, color="#E45756", ls="--", label=f"总体 {final_acc*100:.1f}%")
ax.set_ylim(0, 110)
ax.set_ylabel("准确率 (%)")
ax.set_title("各类别测试准确率")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/05_per_class_acc.png", dpi=120)
plt.close()

# ---------- 产物 ----------
torch.save(model.state_dict(), f"{OUT}/cifar_cnn.pth")
summary = {
    "dataset": "CIFAR-10 (torchvision)",
    "device": gpu_name,
    "torch": torch.__version__,
    "params_m": round(n_param / 1e6, 3),
    "epochs": EPOCHS,
    "final_test_acc": round(final_acc, 4),
    "per_class_acc": {CLASS_ZH[i]: round(per_class[i], 4) for i in range(N_CLASS)},
    "history": hist,
    "seconds": round(time.time() - t0, 1),
}
with open(f"{OUT}/result.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log(f"ARTIFACTS_SAVED {sorted(os.listdir(OUT))}")
log(f"CV_DEMO_DONE acc={final_acc:.4f} seconds={time.time()-t0:.1f} device={gpu_name}")
