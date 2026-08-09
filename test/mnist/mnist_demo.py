"""
MNIST 手写数字识别 — PyTorch CPU 小 Demo
流程: 下载MNIST → 解析idx → 小MLP训练 → 训练曲线+预测可视化(中文)
环境: data-analytics (PyTorch 2.6.0+cpu)
"""
import os
import gzip
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---------- 中文字体 (set_theme 先, 字体后) ----------
sns.set_theme(style="whitegrid", palette="muted")
def setup_chinese_font():
    import glob
    font_files = []
    for pat in ["/usr/share/fonts/**/wqy-microhei.ttc", "/usr/share/fonts/**/wqy-zenhei.ttc",
                "/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/*CJK*.ttc"]:
        font_files += glob.glob(pat, recursive=True)
    for fp in sorted(set(font_files)):
        try: fm.fontManager.addfont(fp)
        except Exception: pass
    zh = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "Noto Sans CJK SC"]
    avail = [n for n in zh if any(f.name == n for f in fm.fontManager.ttflist)]
    if not avail:
        avail = sorted({f.name for f in fm.fontManager.ttflist if "CJK" in f.name or "WenQuanYi" in f.name})
    return avail

zh = setup_chinese_font()
print("可用中文字体:", zh)
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    fm.findfont(fm.FontProperties(family=zh[0]))
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 下载 MNIST ----------
DATA_DIR = "/tmp/mnist_data"
os.makedirs(DATA_DIR, exist_ok=True)
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "train_img",
    "train-labels-idx1-ubyte.gz": "train_lbl",
    "t10k-images-idx3-ubyte.gz": "test_img",
    "t10k-labels-idx1-ubyte.gz": "test_lbl",
}
print("【下载 MNIST 数据集】")
for fn in FILES:
    path = os.path.join(DATA_DIR, fn)
    if not os.path.exists(path):
        print(f"  下载 {fn} ...")
        urllib.request.urlretrieve(BASE + fn, path)
    else:
        print(f"  {fn} 已存在")

def load_idx(path, kind):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16 if "image" in kind else 8)
    if "image" in kind:
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    return data.astype(np.int64)

X_train = load_idx(f"{DATA_DIR}/train-images-idx3-ubyte.gz", "image")
y_train = load_idx(f"{DATA_DIR}/train-labels-idx1-ubyte.gz", "label")
X_test = load_idx(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz", "image")
y_test = load_idx(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz", "label")
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"标签分布: {np.bincount(y_train[:1000]).tolist()} (前1000个)")

# ---------- 2. 构建小 MLP ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转张量 (只用前 12000 训练, 保证 CPU 上快)
X_tr = torch.from_numpy(X_train[:12000])
y_tr = torch.from_numpy(y_train[:12000])
X_te = torch.from_numpy(X_test[:2000])
y_te = torch.from_numpy(y_test[:2000])
BATCH = 256
EPOCHS = 5

print(f"\n【开始训练】 {EPOCHS} epochs, 训练样本 {X_tr.shape[0]}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
train_losses, test_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_tr.shape[0])
    total_loss = 0.0
    nbatch = 0
    for i in range(0, X_tr.shape[0], BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_tr[idx], y_tr[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nbatch += 1
    # 测试准确率
    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1)
        acc = (preds == y_te).float().mean().item()
    avg_loss = total_loss / nbatch
    train_losses.append(avg_loss)
    test_accs.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | 训练损失 {avg_loss:.4f} | 测试准确率 {acc*100:.2f}%")

print(f"\n✅ 最终测试准确率: {test_accs[-1]*100:.2f}%")

# ---------- 3. 可视化 ----------
# 图1: 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, EPOCHS+1), train_losses, marker="o", color="#4C72B0", linewidth=2)
axes[0].set_title("训练损失下降曲线")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("损失 (CrossEntropy)")
axes[1].plot(range(1, EPOCHS+1), [a*100 for a in test_accs], marker="o", color="#55A868", linewidth=2)
axes[1].set_title("测试准确率上升曲线")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("准确率 (%)")
plt.tight_layout()
plt.savefig("/tmp/mnist_training.png", dpi=120)
print("✅ 训练曲线图: /tmp/mnist_training.png")

# 图2: 预测示例 (8x8 网格)
model.eval()
with torch.no_grad():
    preds = model(X_te).argmax(dim=1).numpy()
fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    img = X_te[i].numpy()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"P:{preds[i]} T:{y_te[i].item()}", fontsize=8,
                 color="#2E8B57" if preds[i] == y_te[i].item() else "#C44E52")
    ax.axis("off")
fig.suptitle("MNIST 预测示例 (绿=正确 红=错误)", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/mnist_pred.png", dpi=120)
print("✅ 预测示例图: /tmp/mnist_pred.png")

print("\n" + "=" * 56)
print("MNIST 手写数字识别 Demo 完成!")
print("MNIST_DEMO_OK")
