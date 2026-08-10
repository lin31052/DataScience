#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车牌字符数据集 EDA — 类别分布 → 样本网格 → 像素统计 (CPU)
数据: aladdinss/license-plate-digits-classification-dataset (34 类, 35500 张, 75x100 灰度)
"""
import sys, json, glob, os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from PIL import Image

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
        r = subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                           capture_output=True, text=True, timeout=300)
        log('apt rc=' + str(r.returncode))
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

# ---------- 1. 类别分布 ----------
log("=" * 50); log("1. 类别分布")
cls_dirs = sorted([d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))])
counts = {c: len([f for f in os.listdir(os.path.join(DATA, c)) if os.path.isfile(os.path.join(DATA, c, f))])
          for c in cls_dirs}
total = sum(counts.values())
log(f"类目数: {len(cls_dirs)} | 总样本: {total}")
log("各类: " + ", ".join(f"{c}={counts[c]}" for c in cls_dirs))
report['n_classes'] = len(cls_dirs); report['total'] = int(total)
report['class_counts'] = counts

fig, ax = plt.subplots(figsize=(12, 4.5))
vals = [counts[c] for c in cls_dirs]
colors = ["#e74c3c" if c.isdigit() else "#4C72B0" for c in cls_dirs]
bars = ax.bar(cls_dirs, vals, color=colors, edgecolor="white")
ax.set_title(f"车牌字符类别分布（34 类，共 {total} 张，数字=红 字母=蓝）", fontsize=13)
ax.set_xlabel("字符"); ax.set_ylabel("样本数")
ax.set_ylim(0, max(vals) * 1.15)
plt.tight_layout(); plt.savefig(f"{OUT}/char_eda_01_class_dist.png", dpi=110); plt.close()

# ---------- 2. 样本网格 (0-9 + A, 每类 5 个) ----------
fig, axes = plt.subplots(11, 5, figsize=(9, 14))
show_cls = cls_dirs[:11]
for i, c in enumerate(show_cls):
    files = sorted(os.listdir(os.path.join(DATA, c)))[:5]
    for j, f in enumerate(files):
        ax = axes[i, j]
        img = Image.open(os.path.join(DATA, c, f)).convert('L')
        ax.imshow(img, cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"'{c}'", fontsize=11)
fig.suptitle("车牌字符样本（每类前 11 个）", fontsize=15)
plt.tight_layout(); plt.savefig(f"{OUT}/char_eda_02_samples.png", dpi=110); plt.close()

# ---------- 3. 像素统计 ----------
log("=" * 50); log("2. 像素统计")
sample_files = []
for c in cls_dirs[:6]:
    sample_files += [os.path.join(DATA, c, f) for f in sorted(os.listdir(os.path.join(DATA, c)))[:200]]
arrs = np.stack([np.array(Image.open(f).convert('L')) for f in sample_files])
log(f"图片尺寸: {arrs.shape[2]}x{arrs.shape[1]} | 像素范围 {arrs.min()}-{arrs.max()} | 均值 {arrs.mean():.1f} | 标准差 {arrs.std():.1f}")
report['img_size'] = [int(arrs.shape[2]), int(arrs.shape[1])]
report['pixel_stats'] = {'min': int(arrs.min()), 'max': int(arrs.max()),
                         'mean': round(float(arrs.mean()), 1), 'std': round(float(arrs.std()), 1)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].hist(arrs.ravel(), bins=32, color="#4C72B0", alpha=0.85)
axes[0].set_title("像素灰度直方图", fontsize=12)
axes[0].set_xlabel("灰度值"); axes[0].set_ylabel("像素数")
axes[1].imshow(arrs.mean(axis=0), cmap='gray')
axes[1].set_title("平均字符图像（笔画区域）", fontsize=12)
axes[1].set_xticks([]); axes[1].set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/char_eda_03_pixel_stats.png", dpi=110); plt.close()

with open(f"{OUT}/char_eda_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
log('CHAR_EDA_DONE')
