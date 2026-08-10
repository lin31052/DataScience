# Fashion-MNIST 图像 EDA + 简单 CNN 训练 (T4 GPU)
# 数据: zalando-research/fashionmnist (notebook 内 kaggle CLI 下载)
# 流程: 探测数据格式 → 图像 EDA(类别分布/样本/类别平均图/像素分布) → CNN 训练 → 评估 → 产物
import os, sys, json, time
import numpy as np

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ===== 0. 读数据路径 (runner 写入) =====
with open('/kaggle/working/data_path.txt') as f:
    DATA = f.read().strip()
log('DATA_DIR: ' + DATA)
log('FILES: ' + str(os.listdir(DATA)))

LABELS = ['T恤', '长裤', '套头衫', '连衣裙', '外套',
          '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# ===== 1. 数据加载 (兼容 CSV / 图片目录两种格式) =====
import pandas as pd
from PIL import Image

def find_csv(d):
    for f in sorted(os.listdir(d)):
        if f.endswith('.csv'):
            return os.path.join(d, f)
    return None

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    n = int(round(np.sqrt(X.shape[1])))
    return X.reshape(-1, 1, n, n), y, df.shape

def load_imgs(d):
    Xs, ys = [], []
    for sub in sorted(os.listdir(d)):
        p = os.path.join(d, sub)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(p, f)).convert('L').resize((28, 28))
                Xs.append(np.array(img, dtype=np.float32) / 255.0)
                ys.append(int(sub))
    X = np.stack(Xs)[:, None, :, :]
    return X, np.array(ys), (len(ys), 28, 28)

csv_p = find_csv(DATA)
train_csv = [f for f in (os.listdir(DATA) if os.path.isdir(DATA) else []) if 'train' in f.lower() and f.endswith('.csv')]
test_csv  = [f for f in (os.listdir(DATA) if os.path.isdir(DATA) else []) if 'test' in f.lower() and f.endswith('.csv')]

if train_csv and test_csv:
    log('FORMAT: CSV 模式')
    X_train, y_train, shp = load_csv(os.path.join(DATA, train_csv[0]))
    X_test, y_test, shp2 = load_csv(os.path.join(DATA, test_csv[0]))
    log(f'TRAIN shape={X_train.shape} TEST shape={X_test.shape}')
else:
    log('FORMAT: 图片目录模式')
    X_train, y_train, shp = load_imgs(DATA)
    X_test, y_test, shp2 = X_train, y_train  # 占位, 下方按目录拆分
    # 若 train/ 与 test/ 子目录分开, 分别加载
    for sub in ('train', 'test'):
        p = os.path.join(DATA, sub)
        if os.path.isdir(p):
            if sub == 'train':
                X_train, y_train, shp = load_imgs(p)
            else:
                X_test, y_test, shp2 = load_imgs(p)
    log(f'TRAIN shape={X_train.shape} TEST shape={X_test.shape}')

n_classes = len(np.unique(y_train))
log(f'CLASSES: {n_classes}')

# ===== 2. 中文字体 (铁律17: Kaggle 容器默认无中文字体) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

def setup_chinese_font_kaggle():
    import glob, subprocess
    found = []
    for pat in ["/usr/share/fonts/**/NotoSansCJK*.ttc",
                "/usr/share/fonts/**/*CJK*.ttc",
                "/usr/share/fonts/opentype/noto/*.ttc",
                "/usr/share/fonts/**/wqy*.ttc"]:
        found += glob.glob(pat, recursive=True)
    if not found:
        log('安装 fonts-noto-cjk ...')
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        r = subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                           capture_output=True, text=True, timeout=300)
        log('apt rc=' + str(r.returncode))
        found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + \
                glob.glob("/usr/share/fonts/opentype/noto/*.ttc", recursive=True)
    for fp in sorted(set(found)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    return sorted({f.name for f in fm.fontManager.ttflist
                   if any(k in f.name for k in ("CJK", "WenQuanYi"))})

zh = setup_chinese_font_kaggle()
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
log('FONT_OK: ' + str(zh))

# ===== 3. 图像 EDA =====
outdir = '/kaggle/working/out'
os.makedirs(outdir, exist_ok=True)
log('STEP_EDA')

# 3.1 类别分布
fig, ax = plt.subplots(figsize=(9, 4.5))
counts = pd.Series(y_train).value_counts().sort_index()
bars = ax.bar(range(n_classes), counts.values, color=sns.color_palette("husl", n_classes))
ax.set_xticks(range(n_classes))
ax.set_xticklabels(LABELS[:n_classes], rotation=30)
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, str(v), ha='center', fontsize=9)
ax.set_title(f'Fashion-MNIST 训练集类别分布 (共 {len(y_train)} 张)')
ax.set_ylabel('样本数')
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_eda_class_dist.png', dpi=120)
plt.close()

# 3.2 每类样本展示 (每类 5 张)
fig, axes = plt.subplots(n_classes, 5, figsize=(7, 12))
for c in range(n_classes):
    idx = np.where(y_train == c)[0][:5]
    for j, i in enumerate(idx):
        axes[c, j].imshow(X_train[i, 0], cmap='gray')
        axes[c, j].axis('off')
        if j == 0:
            axes[c, j].set_ylabel(LABELS[c], fontsize=9)
fig.suptitle('Fashion-MNIST 每类样本示例', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_eda_samples.png', dpi=120)
plt.close()

# 3.3 类别平均图 (10 个 28x28 平均图像, 直观展示每类"平均外观")
fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
for c in range(n_classes):
    avg = X_train[y_train == c].mean(axis=0)[0]
    ax = axes[c // 5, c % 5]
    im = ax.imshow(avg, cmap='viridis')
    ax.set_title(f'{LABELS[c]}\n均值强度 {avg.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('Fashion-MNIST 各类别平均图像 (28x28)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_eda_class_avg.png', dpi=120)
plt.close()

# 3.4 像素强度分布
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(X_train.ravel(), bins=50, color='#4C72B0', alpha=0.8)
ax.set_xlabel('像素强度 (0-1)')
ax.set_ylabel('像素数量')
ax.set_title(f'Fashion-MNIST 像素强度分布 (共 {X_train.shape[0]*28*28:,} 像素)')
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_eda_pixels.png', dpi=120)
plt.close()
log('EDA_IMAGES_SAVED: ' + str(sorted(os.listdir(outdir))))

# ===== 4. CNN 训练 (GPU) =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

log('STEP_TRAIN')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'torch {torch.__version__} | device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN().to(device)
n_params = sum(p.numel() for p in model.parameters())
log(f'MODEL: SimpleCNN, params={n_params:,}')

X_t = torch.tensor(X_train)
y_t = torch.tensor(y_train)
X_e = torch.tensor(X_test)
y_e = torch.tensor(y_test)
train_ds = TensorDataset(X_t, y_t)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_ds = TensorDataset(X_e, y_e)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
EPOCHS = 5
hist = {'loss': [], 'acc': []}

t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    train_loss = tot_loss / total
    train_acc = correct / total
    # 每 epoch 后测 test acc
    model.eval()
    with torch.no_grad():
        tc, tt = 0, 0
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            tc += (model(xb).argmax(1) == yb).sum().item()
            tt += xb.size(0)
    test_acc = tc / tt
    hist['loss'].append(train_loss)
    hist['acc'].append(test_acc)
    log(f'Epoch {ep}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}')

train_sec = time.time() - t0
log(f'TRAIN_DONE seconds={train_sec:.1f}')

# ===== 5. 评估: 混淆矩阵 + 每类 acc =====
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        all_preds.append(model(xb).argmax(1).cpu().numpy())
        all_true.append(yb.numpy())
all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(all_true, all_preds)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
test_acc_final = float((all_preds == all_true).mean())

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=LABELS[:n_classes], yticklabels=LABELS[:n_classes], ax=ax)
ax.set_xlabel('预测类别')
ax.set_ylabel('真实类别')
ax.set_title(f'Fashion-MNIST 混淆矩阵 (test acc={test_acc_final:.4f})')
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_confusion.png', dpi=120)
plt.close()

# 训练曲线
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(range(1, EPOCHS + 1), hist['loss'], 'o-', color='#C44E52', label='训练损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('损失', color='#C44E52')
ax1.tick_params(axis='y', labelcolor='#C44E52')
ax2 = ax1.twinx()
ax2.plot(range(1, EPOCHS + 1), hist['acc'], 's-', color='#4C72B0', label='测试准确率')
ax2.set_ylabel('测试准确率', color='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#4C72B0')
ax2.set_ylim(0, 1)
ax1.set_title(f'Fashion-MNIST CNN 训练曲线 (T4, {train_sec:.0f}s)')
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_train_curve.png', dpi=120)
plt.close()

# 预测示例 (每类 3 张: 对/错标注)
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
shown = {c: 0 for c in range(n_classes)}
rng = np.random.RandomState(42)
order = rng.permutation(len(all_true))
for i in order:
    c = int(all_true[i])
    if shown[c] >= 3:
        continue
    ok = all_preds[i] == c
    axes[c // 3, (c % 3) * 2 if shown[c] == 0 else (c % 3) * 2 + 1].imshow(X_test[i, 0], cmap='gray')
    ax = axes[c // 3, (c % 3) * 2 if shown[c] == 0 else (c % 3) * 2 + 1]
    ax.set_title(('✓' if ok else f'✗→{LABELS[all_preds[i]]}'), fontsize=9,
                 color='green' if ok else 'red')
    ax.axis('off')
    shown[c] += 1
    if all(v >= 3 for v in shown.values()):
        break
fig.suptitle('Fashion-MNIST 预测示例 (✓=正确 ✗→=误判为)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/fashion_pred_examples.png', dpi=120)
plt.close()

# ===== 6. 汇总 json =====
result = {
    'dataset': 'zalando-research/fashionmnist',
    'task': 'Fashion-MNIST 10类服饰图像分类',
    'data_shape': {'train': list(X_train.shape), 'test': list(X_test.shape)},
    'device': str(device),
    'model': 'SimpleCNN (2xConv3x3-64ch + 2xFC)',
    'params': n_params,
    'epochs': EPOCHS,
    'train_seconds': round(train_sec, 2),
    'test_acc': round(test_acc_final, 4),
    'per_class_acc': {LABELS[i]: round(float(v), 4) for i, v in enumerate(per_class_acc)},
    'train_loss_history': [round(x, 4) for x in hist['loss']],
    'test_acc_history': [round(x, 4) for x in hist['acc']],
}
with open(f'{outdir}/fashion_result.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

log('ARTIFACTS: ' + str(sorted(os.listdir(outdir))))
log('FINAL_TEST_ACC=' + str(round(test_acc_final, 4)))
log('ALL_DONE')
