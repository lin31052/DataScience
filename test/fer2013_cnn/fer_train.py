# FER-2013 表情分类 — 简单 CNN 训练 (Kaggle T4)
# 用法: run_train(data_root, outdir)
# 输出: train_curve.png / confusion.png / result.json
import os, sys, json, time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABELS_ZH = ["愤怒", "厌恶", "恐惧", "开心", "中性", "悲伤", "惊讶"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

def setup_font():
    import glob, subprocess
    found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + \
            glob.glob("/usr/share/fonts/opentype/noto/*.ttc", recursive=True)
    if not found:
        print("中文字体缺失, apt 安装 fonts-noto-cjk ...", file=sys.stderr, flush=True)
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                       capture_output=True, timeout=300)
        found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True)
    for fp in sorted(set(found)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    zh = sorted({f.name for f in fm.fontManager.ttflist if "CJK" in f.name})
    if zh:
        plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

class FerDataset(Dataset):
    def __init__(self, root, split):
        self.paths, self.labels = [], []
        for c in CLASSES:
            d = os.path.join(root, split, c)
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.getsize(p) > 0:
                    self.paths.append(p)
                    self.labels.append(CLS2IDX[c])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("L"), dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0), self.labels[i]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 48->24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 24->12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),# 12->6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def run_train(data_root, outdir, epochs=15, batch_size=128, lr=1e-3):
    os.makedirs(outdir, exist_ok=True)
    setup_font()
    print(f"torch {torch.__version__} | device: "
          f"{'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          file=sys.stderr, flush=True)

    tr_ds = FerDataset(data_root, "train")
    te_ds = FerDataset(data_root, "test")
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"train={len(tr_ds)} test={len(te_ds)}", file=sys.stderr, flush=True)

    model = SimpleCNN().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}", file=sys.stderr, flush=True)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    hist = {"train_loss": [], "train_acc": [], "test_acc": []}
    best = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, nb = 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
            tc += (model(xb).argmax(1) == yb).sum().item()
            nb += len(xb)
        # 验证
        model.eval()
        ta = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                ta += (model(xb.cuda()).argmax(1) == yb.cuda()).sum().item()
        tr_acc = tc / nb
        te_acc = ta / len(te_ds)
        hist["train_loss"].append(tl / nb)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        best = max(best, te_acc)
        print(f"Epoch {ep}/{epochs} loss={tl/nb:.4f} train_acc={tr_acc:.4f} "
              f"test_acc={te_acc:.4f}", file=sys.stderr, flush=True)
    train_sec = time.time() - t0

    # 混淆矩阵
    model.eval()
    cm = np.zeros((7, 7), dtype=int)
    with torch.no_grad():
        for xb, yb in te_dl:
            pred = model(xb.cuda()).argmax(1).cpu().numpy()
            for p, t in zip(pred, yb.numpy()):
                cm[t, p] += 1
    per_acc = {CLASSES[i]: float(cm[i, i] / cm[i].sum()) for i in range(7)}

    # ---- 图: 训练曲线 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(hist["train_loss"], marker="o", color="#C44E52")
    ax1.set_title("训练损失"); ax1.set_xlabel("epoch"); ax1.grid(alpha=0.3)
    ax2.plot(hist["train_acc"], marker="o", color="#4C72B0", label="训练集")
    ax2.plot(hist["test_acc"], marker="s", color="#55A868", label="测试集")
    ax2.set_title(f"准确率 (最佳测试 {best*100:.1f}%)"); ax2.set_xlabel("epoch")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{outdir}/train_curve.png", dpi=120); plt.close()

    # ---- 图: 混淆矩阵 ----
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(LABELS_ZH, fontsize=10)
    ax.set_yticks(range(7)); ax.set_yticklabels(LABELS_ZH, fontsize=10)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_title(f"FER-2013 混淆矩阵 (测试集, 共 {cm.sum():,} 张)\n"
                 f"总准确率 {cm.trace()/cm.sum()*100:.1f}%")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(f"{outdir}/confusion.png", dpi=120); plt.close()

    res = {"test_acc": float(cm.trace() / cm.sum()),
           "best_test_acc": float(best),
           "per_class_acc": per_acc,
           "params": n_params,
           "train_seconds": round(train_sec, 1),
           "epochs": epochs,
           "note": "人类水平 ~65%, FER-2013 出名难: 表情边缘模糊/光照杂乱/主观标注"}
    json.dump(res, open(f"{outdir}/result.json", "w"), ensure_ascii=False, indent=2)
    print(f"TRAIN_DONE test_acc={res['test_acc']:.4f} best={best:.4f} "
          f"seconds={train_sec:.1f}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    run_train(sys.argv[1], sys.argv[2])
