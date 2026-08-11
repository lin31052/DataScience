# -*- coding: utf-8 -*-
"""
CIFAR-10 经典 CNN 架构对比: LeNet-5 (1998) vs AlexNet (2012) vs ResNet18 (2015)
教学点: 从 6 万参数到 1100 万参数, 架构演化如何提升图像分类能力
- 同一数据 / 同一训练配置 / 各 10 epochs, 公平对比
- 产物: /kaggle/working/out/ 下 5 张对比图 + result.json
"""
import os, json, sys, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

OUT = '/kaggle/working/out'
os.makedirs(OUT, exist_ok=True)

# ================= 中文字体 (Kaggle 容器无中文字体, 需安装) =================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
if not any('CJK' in f.name for f in font_manager.fontManager.ttflist):
    os.system('apt-get install -y fonts-noto-cjk > /dev/null 2>&1')
    for f in glob.glob('/usr/share/fonts/**/*.tt[fc]', recursive=True):
        try: font_manager.fontManager.addfont(f)
        except Exception: pass
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_ZH = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']


# ================= 数据加载 (CIFAR-10 python 官方格式) =================
def unpickle(path):
    import pickle
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d


class CIFAR10NP(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        if train:
            files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
        else:
            files = [os.path.join(data_dir, 'test_batch')]
        xs, ys = [], []
        for fp in files:
            d = unpickle(fp)
            xs.append(d[b'data'])
            ys.extend(d[b'labels'])
        self.x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.array(ys)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.transform is not None:
            x = self.transform(torch.from_numpy(x))
        return x, self.y[i]


def get_transforms():
    # CIFAR 标准增强: 随机裁剪+翻转, 测试集仅归一化
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_t = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Normalize(mean, std),
    ])
    test_t = T.Normalize(mean, std)
    return train_t, test_t


# ================= 模型 =================
class LeNet5(nn.Module):
    """LeNet-5 (LeCun 1998), CIFAR 适配: 输入 3x32x32, ~6 万参数"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),      # 32->28->14
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),     # 14->10->5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


class AlexNetCIFAR(nn.Module):
    """AlexNet (2012) 架构适配 CIFAR-10 (32x32): 5 conv + 3 fc
    原版 conv1 11x11/s4 + 3 个 maxpool 会把 32x32 压成 1x1, 必须适配:
    改用 3x3 conv + 2x2 maxpool, 保留 AlexNet 的深卷积+大 FC 风格 (~36M 参数)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 16->8
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 8->4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 4 * 4, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


def make_model(name):
    if name == 'lenet5':
        return LeNet5()
    if name == 'alexnet':
        return AlexNetCIFAR()
    if name == 'resnet18':
        return torchvision.models.resnet18(weights=None, num_classes=10)
    raise ValueError(name)


# ================= 训练 =================
def train_one_model(name, train_loader, test_loader, epochs=10, device='cuda'):
    model = make_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    t0 = time.time()
    print(f'\n===== 训练 {name} | 参数 {n_params/1e6:.2f}M =====', flush=True)
    for ep in range(epochs):
        model.train()
        tot_l, tot_c, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot_l += loss.item() * y.size(0)
            tot_c += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        sched.step()
        tr_acc = tot_c / n
        te_acc = evaluate(model, test_loader, device)
        history['train_loss'].append(tot_l / n)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        print(f'  epoch {ep+1}/{epochs} | train_loss {tot_l/n:.4f} | train_acc {tr_acc:.4f} | test_acc {te_acc:.4f}',
              flush=True)
    train_sec = time.time() - t0
    return model, n_params, history, train_sec


@torch.no_grad()
def evaluate(model, loader, device='cuda'):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def class_accuracy_and_confusion(model, loader, device='cuda'):
    model.eval()
    n_classes = 10
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p] += 1
    acc = conf.diagonal() / conf.sum(1)
    return acc, conf


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device} | torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if device=="cuda" else "无"}', flush=True)

    # 数据路径: /kaggle/working/kdata/cifar-10-batches-py/
    cands = glob.glob('/kaggle/working/kdata/*/cifar-10-batches-py') + glob.glob('/kaggle/working/kdata/cifar-10-batches-py')
    data_dir = cands[0] if cands else None
    if data_dir is None:
        # 直接解压在 kdata 下
        batch = glob.glob('/kaggle/working/kdata/**/data_batch_1', recursive=True)
        if batch:
            data_dir = os.path.dirname(batch[0])
    assert data_dir, '找不到 CIFAR-10 数据! 请检查解压路径'
    print('数据目录:', data_dir, flush=True)

    train_t, test_t = get_transforms()
    train_ds = CIFAR10NP(data_dir, train=True, transform=train_t)
    test_ds = CIFAR10NP(data_dir, train=False, transform=test_t)
    print(f'训练集 {len(train_ds)} / 测试集 {len(test_ds)}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    names = ['lenet5', 'alexnet', 'resnet18']
    labels = ['LeNet-5 (1998)', 'AlexNet (2012)', 'ResNet18 (2015)']
    only = os.environ.get('CIFAR_ONLY_MODEL')
    if only:
        names = [only]
        labels = [labels[['lenet5', 'alexnet', 'resnet18'].index(only)]]
        print(f'调试模式: 只训练 {only}', flush=True)
    results = {}
    curves = {}
    for name, lab in zip(names, labels):
        model, n_params, hist, sec = train_one_model(name, train_loader, test_loader, epochs=10, device=device)
        acc, conf = class_accuracy_and_confusion(model, test_loader, device)
        results[name] = {
            'label': lab, 'params': n_params, 'train_sec': sec,
            'test_acc': hist['test_acc'][-1], 'best_test_acc': max(hist['test_acc']),
            'best_epoch': int(np.argmax(hist['test_acc'])) + 1,
            'class_acc': {CLASSES[i]: float(acc[i]) for i in range(10)},
            'confusion': conf.tolist(),
        }
        curves[name] = hist
        torch.cuda.empty_cache()
        print(f'{lab} 完成: test_acc={hist["test_acc"][-1]:.4f} 用时 {sec:.0f}s', flush=True)

    # ============ 画图 ============
    colors = {'lenet5': '#4C72B0', 'alexnet': '#DD8452', 'resnet18': '#55A868'}

    # 1) 训练/测试准确率曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, lab) in zip(axes, zip(names, labels)):
        h = curves[name]
        ep = np.arange(1, len(h['train_acc']) + 1)
        ax.plot(ep, np.array(h['train_acc']) * 100, 'o-', color=colors[name], label='训练')
        ax.plot(ep, np.array(h['test_acc']) * 100, 's--', color='#C44E52', label='测试')
        ax.set_title(lab, fontsize=12)
        ax.set_xlabel('epoch'); ax.set_ylabel('准确率 (%)')
        ax.set_ylim(0, 100); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('CIFAR-10 三架构训练曲线 (各 10 epochs, SGD lr=0.01 cosine)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{OUT}/curve_compare.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 2) 测试准确率柱状图 + 参数量
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    accs = [results[n]['test_acc'] * 100 for n in names]
    bars = ax.bar(labels, accs, color=[colors[n] for n in names], width=0.55)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.8, f'{a:.1f}%', ha='center', fontsize=11)
    ax.set_ylabel('测试准确率 (%)'); ax.set_title('测试集准确率对比 (10 epochs)', fontsize=12)
    ax.set_ylim(0, max(accs) * 1.18); ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    params = [results[n]['params'] / 1e6 for n in names]
    bars = ax.bar(labels, params, color=[colors[n] for n in names], width=0.55)
    for b, p in zip(bars, params):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.3, f'{p:.1f}M', ha='center', fontsize=11)
    ax.set_ylabel('参数量 (百万)'); ax.set_title('模型参数量对比', fontsize=12)
    ax.set_ylim(0, max(params) * 1.18); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/accuracy_params.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 3) 每类准确率
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(10); w = 0.27
    for i, n in enumerate(names):
        ca = [results[n]['class_acc'][c] * 100 for c in CLASSES]
        ax.bar(x + (i - 1) * w, ca, w, label=labels[i], color=colors[n])
    ax.set_xticks(x); ax.set_xticklabels(CLASS_ZH, fontsize=10)
    ax.set_ylabel('准确率 (%)'); ax.set_title('各类别准确率对比', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f'{OUT}/class_acc.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 4) 混淆矩阵 1x3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, n, lab in zip(axes, names, labels):
        conf = np.array(results[n]['confusion'])
        conf_n = conf / conf.sum(1, keepdims=True)
        im = ax.imshow(conf_n, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_ZH, fontsize=7, rotation=45)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_ZH, fontsize=7)
        ax.set_title(f'{lab} (acc {results[n]["test_acc"]*100:.1f}%)', fontsize=11)
        for i in range(10):
            for j in range(10):
                if conf_n[i, j] > 0.5:
                    ax.text(j, i, f'{conf_n[i,j]:.2f}', ha='center', va='center', fontsize=5.5, color='white')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('CIFAR-10 三架构混淆矩阵 (归一化)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{OUT}/confusion_compare.png', dpi=130, bbox_inches='tight')
    plt.close()

    # ============ 汇总 json ============
    summary = {
        'task': 'CIFAR-10 经典 CNN 架构对比 (LeNet-5 vs AlexNet vs ResNet18)',
        'data': 'pankrzysiu/cifar10-python', 'epochs': 10,
        'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'models': {n: {k: v for k, v in results[n].items() if k != 'confusion'} for n in names},
    }
    with open(f'{OUT}/result.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n========== 最终结果 ==========')
    for n in names:
        r = results[n]
        print(f"{r['label']}: 参数 {r['params']/1e6:.2f}M | test_acc {r['test_acc']*100:.2f}% "
              f"(best {r['best_test_acc']*100:.2f}% @ ep{r['best_epoch']}) | 训练 {r['train_sec']:.0f}s")
    print('ALL_DONE')


if __name__ == '__main__':
    main()
