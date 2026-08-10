# Fashion-MNIST 图像分类: ResNet50 vs ResNet50+SE vs ResNet50+CBAM 注意力对比 (T4 x2 双卡并行, ~10min)
# 数据: zalando-research/fashionmnist (kaggle CLI 下载, CSV: label + 784 像素)
# 流程: 三模型(裸/SE通道/CBAM通道+空间)交替 step 双卡同时训练各 15 epochs → 对比曲线/混淆矩阵
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

# ===== 2. 中文字体 (铁律17) =====
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

# ===== 3. 模型: ResNet50 小图版 (CIFAR 风格 stem) + 可选 SE/CBAM 注意力 =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

n_gpu = torch.cuda.device_count()
log(f'torch {torch.__version__} | gpu_count={n_gpu} | '
    + ' | '.join(torch.cuda.get_device_name(i) for i in range(n_gpu)))
if n_gpu == 0:
    log('ERROR: no GPU!'); sys.exit(1)

class SELayer(nn.Module):  # SE 通道注意力 (avg 池化)
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class ChannelAttention(nn.Module):  # CBAM 通道注意力 (avg+max 双路)
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels))
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx  = self.mlp(self.max_pool(x).view(b, c))
        w = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * w

class SpatialAttention(nn.Module):  # CBAM 空间注意力
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, mx], dim=1)
        w = torch.sigmoid(self.conv(cat))
        return x * w

class CBAM(nn.Module):  # 通道注意力 → 空间注意力 串联
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride=1, attn=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * 4))
        ch = out_ch * 4
        self.attn = {'se': SELayer(ch, 16), 'cbam': CBAM(ch, 16)}.get(attn, nn.Identity())
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)   # 残差相加
        out = self.attn(out)              # 注意力统一放相加后 (SE/CBAM 同位置公平对比)
        return out

class ResNet50(nn.Module):  # 小图版: 3x3 stride1 stem, 无 maxpool (CIFAR 风格)
    def __init__(self, attn=None, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make(64, 64, 3, 1, attn)
        self.layer2 = self._make(64, 128, 4, 2, attn)
        self.layer3 = self._make(128, 256, 6, 2, attn)
        self.layer4 = self._make(256, 512, 3, 2, attn)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, num_classes)
    def _make(self, in_ch, out_ch, n, stride, attn):
        layers = [Bottleneck(in_ch, out_ch, stride, attn)]
        for _ in range(1, n):
            layers.append(Bottleneck(out_ch * 4, out_ch, 1, attn))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                      batch_size=128, shuffle=True)
test_dl  = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                      batch_size=256)

# ===== 4. 三模型双卡并行训练 (交替 step) =====
log('STEP_TRAIN')
TAGS = ['ResNet50', 'ResNet50+SE', 'ResNet50+CBAM']
ATTNS = [None, 'se', 'cbam']
EPOCHS = 15

# 设备分配: 卡0 放 1 个, 卡1 放 2 个 (均衡)
devs = ['cuda:0', 'cuda:1', 'cuda:1'] if n_gpu >= 2 else ['cuda:0'] * 3

models, opts, infos = [], [], []
for i, (attn, tag) in enumerate(zip(ATTNS, TAGS)):
    torch.manual_seed(42)
    m = ResNet50(attn=attn).to(devs[i])
    m.tag = tag
    np_ = sum(p.numel() for p in m.parameters())
    opts.append(torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4))
    models.append(m); infos.append({'tag': tag, 'params': np_, 'dev': devs[i]})
    log(f'MODEL {tag}: params={np_:,} device={devs[i]}')

hists = [{'loss': [], 'acc': []} for _ in models]
t_start = time.time()
for ep in range(1, EPOCHS + 1):
    iters = [iter(train_dl) for _ in models]
    tots = [0] * 3; corrs = [0] * 3; losses = [0.0] * 3; t0 = time.time()
    for it in range(len(train_dl)):
        for i, (m, dev) in enumerate(zip(models, devs)):
            xb, yb = next(iters[i])
            xb, yb = xb.to(dev), yb.to(dev)
            opts[i].zero_grad()
            out = m(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward(); opts[i].step()
            losses[i] = loss.item()
            tots[i] += yb.size(0); corrs[i] += (out.argmax(1) == yb).sum().item()
    line = []
    for i, m in enumerate(models):
        acc = corrs[i] / tots[i]
        hists[i]['loss'].append(losses[i]); hists[i]['acc'].append(acc)
        line.append(f'{m.tag} {losses[i]:.4f}/{acc:.4f}')
    log(f'epoch {ep}/{EPOCHS} | ' + ' | '.join(line) + f' ({time.time()-t0:.0f}s)')

# ===== 5. 测试集评估 =====
log('STEP_EVAL')
preds_all = []
for i, (m, dev) in enumerate(zip(models, devs)):
    m.eval()
    with torch.no_grad():
        preds = torch.cat([m(xb.to(dev)).argmax(1).cpu() for xb, _ in test_dl]).numpy()
    test_acc = (preds == y_test).mean().item()
    infos[i]['test_acc'] = test_acc
    preds_all.append(preds)
    log(f'RESULT {infos[i]["tag"]}: test_acc={test_acc:.4f}')

# ===== 6. 对比图 + 混淆矩阵 =====
log('STEP_VIS')
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, key in zip(axes, ['loss', 'acc']):
    for i, tag in enumerate(TAGS):
        ax.plot(range(1, EPOCHS + 1), hists[i][key],
                marker='o', label=f"{tag} (测试 {infos[i]['test_acc']*100:.2f}%)")
    ax.set_xlabel('epoch'); ax.set_ylabel('loss' if key == 'loss' else '训练准确率')
    ax.set_title('训练 ' + ('损失' if key == 'loss' else '准确率') + ' 曲线对比')
    ax.legend(fontsize=8)
plt.suptitle(f'Fashion-MNIST: ResNet50 vs +SE vs +CBAM 注意力 (各 {EPOCHS} epochs, T4 x2 并行)', fontsize=13)
plt.tight_layout(); plt.savefig(f'{outdir}/rn50_curve_compare.png', dpi=120); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
for ax, tag, preds in zip(axes, TAGS, preds_all):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    ax.set_title(f"{tag} 混淆矩阵 (测试 {infos[i]['test_acc']*100:.2f}%)")
    ax.set_xlabel('预测'); ax.set_ylabel('真实')
plt.tight_layout(); plt.savefig(f'{outdir}/rn50_confusion_compare.png', dpi=120); plt.close()

summary = {t: {'test_acc': round(r['test_acc'], 4), 'params': r['params'], 'device': r['dev']}
           for t, r in zip(TAGS, infos)}
best = max(TAGS, key=lambda t: summary[t]['test_acc'])
summary['结论'] = f"最佳: {best} ({summary[best]['test_acc']*100:.2f}%); 总训练时长 {time.time()-t_start:.0f}s; " + '; '.join(
    f"{a} vs {b} {((summary[b]['test_acc']-summary[a]['test_acc'])*100):+.2f}pp"
    for a, b in [('ResNet50', 'ResNet50+SE'), ('ResNet50', 'ResNet50+CBAM'), ('ResNet50+SE', 'ResNet50+CBAM')])
with open(f'{outdir}/rn50_result.json', 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log('ARTIFACTS_SAVED: ' + str(sorted(os.listdir(outdir))))
log('TRAIN_DONE ' + json.dumps(summary, ensure_ascii=False))