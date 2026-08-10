# Fashion-MNIST 图像分类: 简单 CNN vs CNN+SE注意力 对比实验 (T4 GPU, ~2min)
# 数据: zalando-research/fashionmnist (kaggle CLI 下载, CSV: label + 784 像素)
# 流程: 轻量 EDA → 两个相同结构模型(开/关 SE 通道注意力)各训 10 epochs → 对比曲线/混淆矩阵
import os, sys, json, time
import numpy as np

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ===== 0. 读数据路径 (runner 写入) =====
with open('/kaggle/working/data_path.txt') as f:
    DATA = f.read().strip()
log('DATA_DIR: ' + DATA + ' FILES: ' + str(os.listdir(DATA)))

LABELS = ['T恤', '长裤', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# ===== 1. 数据加载 (CSV 模式) =====
import pandas as pd

def load_csv(p):
    df = pd.read_csv(p)
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    return X.reshape(-1, 1, 28, 28), y

csvs = sorted(f for f in os.listdir(DATA) if f.endswith('.csv'))
train_csv = [f for f in csvs if 'train' in f.lower()][0]
test_csv  = [f for f in csvs if 'test' in f.lower()][0]
X_train, y_train = load_csv(os.path.join(DATA, train_csv))
X_test,  y_test  = load_csv(os.path.join(DATA, test_csv))
log(f'LOADED: train={X_train.shape} test={X_test.shape} classes={len(np.unique(y_train))}')

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
                "/usr/share/fonts/opentype/noto/*.ttc"]:
        found += glob.glob(pat, recursive=True)
    if not found:
        log('installing fonts-noto-cjk ...')
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                       capture_output=True, text=True, timeout=300)
        found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True)
    for fp in sorted(set(found)):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    return sorted({f.name for f in fm.fontManager.ttflist if 'CJK' in f.name})

zh = setup_chinese_font_kaggle()
if zh:
    plt.rcParams["font.sans-serif"] = zh + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
log('FONT_OK: ' + str(zh))

outdir = '/kaggle/working/out'
os.makedirs(outdir, exist_ok=True)

# ===== 3. 轻量 EDA (2 张图) =====
log('STEP_EDA')
fig, ax = plt.subplots(figsize=(9, 4.2))
counts = pd.Series(y_train).value_counts().sort_index()
bars = ax.bar(range(10), counts.values, color=sns.color_palette("husl", 10))
ax.set_xticks(range(10)); ax.set_xticklabels(LABELS, rotation=30)
for i, v in enumerate(counts.values):
    ax.text(i, v + 300, str(v), ha='center', fontsize=9)
ax.set_title(f'Fashion-MNIST 训练集类别分布 (共 {len(y_train):,} 张 28x28 灰度图)')
plt.tight_layout(); plt.savefig(f'{outdir}/se_eda_class_dist.png', dpi=120); plt.close()

fig, axes = plt.subplots(10, 5, figsize=(6.5, 11))
for c in range(10):
    idx = np.where(y_train == c)[0][:5]
    for j, i in enumerate(idx):
        axes[c, j].imshow(X_train[i, 0], cmap='gray'); axes[c, j].axis('off')
        if j == 0: axes[c, j].set_ylabel(LABELS[c], fontsize=9)
fig.suptitle('Fashion-MNIST 每类样本示例', fontsize=13)
plt.tight_layout(); plt.savefig(f'{outdir}/se_eda_samples.png', dpi=120); plt.close()
log('EDA_IMAGES_SAVED')

# ===== 4. 模型: 相同简单 CNN, 可选 SE 通道注意力 =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'torch {torch.__version__} | device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))

class SELayer(nn.Module):  # Squeeze-and-Excitation 通道注意力
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)      # 全局信息压缩
        w = self.fc(w).view(b, c, 1, 1)      # 学习各通道重要度
        return x * w                         # 通道加权

class SimpleCNN(nn.Module):
    def __init__(self, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.se = SELayer(64) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                      batch_size=128, shuffle=True)
test_dl  = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                      batch_size=256)

def train_one(model, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'loss': [], 'acc': []}
    for ep in range(1, epochs + 1):
        model.train(); tot, corr, t0 = 0, 0, time.time()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            tot += yb.size(0); corr += (model(xb).argmax(1) == yb).sum().item()
        acc = corr / tot
        hist['loss'].append(loss.item()); hist['acc'].append(acc)
        log(f'  {model.__class__.__name__}[{model.tag}] epoch {ep}/{epochs} loss {loss.item():.4f} train_acc {acc:.4f} ({time.time()-t0:.1f}s)')
    model.eval()
    preds = torch.cat([model(xb.to(device)).argmax(1).cpu() for xb, _ in test_dl])
    test_acc = (preds == torch.from_numpy(y_test)).float().mean().item()
    return hist, preds.numpy(), test_acc

# ===== 5. 训练两个模型 (开/关 SE) =====
log('STEP_TRAIN')
results = {}
for use_se, tag in [(False, 'CNN'), (True, 'CNN+SE')]:
    torch.manual_seed(42)
    model = SimpleCNN(use_se=use_se).to(device)
    model.tag = tag
    n_params = sum(p.numel() for p in model.parameters())
    log(f'MODEL {tag}: params={n_params:,}')
    hist, preds, test_acc = train_one(model)
    results[tag] = {'use_se': use_se, 'params': n_params, 'test_acc': test_acc,
                    'hist': hist, 'preds': preds}
    log(f'RESULT {tag}: test_acc={test_acc:.4f}')

# ===== 6. 对比图 + 混淆矩阵 =====
log('STEP_VIS')
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, key in zip(axes, ['loss', 'acc']):
    for tag in results:
        ax.plot(range(1, 11), results[tag]['hist'][key],
                marker='o', label=f"{tag} (测试 {results[tag]['test_acc']*100:.2f}%)")
    ax.set_xlabel('epoch'); ax.set_ylabel('loss' if key == 'loss' else '训练准确率')
    ax.set_title('训练 ' + ('损失' if key == 'loss' else '准确率') + ' 曲线对比')
    ax.legend()
plt.suptitle('Fashion-MNIST: 简单 CNN vs CNN+SE 注意力 (各 10 epochs)', fontsize=13)
plt.tight_layout(); plt.savefig(f'{outdir}/se_curve_compare.png', dpi=120); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, tag in zip(axes, results):
    cm = confusion_matrix(y_test, results[tag]['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    ax.set_title(f"{tag} 混淆矩阵 (测试准确率 {results[tag]['test_acc']*100:.2f}%)")
    ax.set_xlabel('预测'); ax.set_ylabel('真实')
plt.tight_layout(); plt.savefig(f'{outdir}/se_confusion_compare.png', dpi=120); plt.close()

summary = {tag: {'test_acc': round(r['test_acc'], 4), 'params': r['params']} for tag, r in results.items()}
summary['结论'] = ('SE 注意力有效, 提升 ' + f"{(results['CNN+SE']['test_acc']-results['CNN']['test_acc'])*100:.2f} 个百分点"
                  if results['CNN+SE']['test_acc'] > results['CNN']['test_acc'] else 'SE 注意力无明显提升')
with open(f'{outdir}/se_result.json', 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log('ARTIFACTS_SAVED: ' + str(sorted(os.listdir(outdir))))
log('TRAIN_DONE ' + json.dumps(summary, ensure_ascii=False))
