# FER-2013 人脸表情 EDA — 数据统计规律直接可视化
# 用法: run_eda(data_root, outdir)
# 输出: 01_class_dist.png 类别分布 / 02_mean_faces.png 平均脸 /
#       03_class_brightness.png 亮度统计 / fer_eda_stats.json
import os, sys, json
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABELS_ZH = ["愤怒", "厌恶", "恐惧", "开心", "中性", "悲伤", "惊讶"]

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
    return zh

def load_class_images(data_root, split, max_per_class=None):
    """返回 {cls: np.array (N,48,48) uint8}"""
    out = {}
    for c in CLASSES:
        d = os.path.join(data_root, split, c)
        files = [f for f in os.listdir(d) if os.path.getsize(os.path.join(d, f)) > 0]
        if max_per_class:
            files = files[:max_per_class]
        arr = np.zeros((len(files), 48, 48), dtype=np.uint8)
        for i, f in enumerate(files):
            arr[i] = np.array(Image.open(os.path.join(d, f)).convert("L"))
        out[c] = arr
    return out

def run_eda(data_root, outdir):
    os.makedirs(outdir, exist_ok=True)
    zh = setup_font()
    print("EDA 字体:", zh, file=sys.stderr, flush=True)

    tr = load_class_images(data_root, "train")
    te = load_class_images(data_root, "test")
    n_tr = {c: len(v) for c, v in tr.items()}
    n_te = {c: len(v) for c, v in te.items()}
    n_all = {c: n_tr[c] + n_te[c] for c in CLASSES}
    total = sum(n_all.values())
    print(f"train={sum(n_tr.values())} test={sum(n_te.values())} total={total}",
          file=sys.stderr, flush=True)

    stats = {"total": total,
             "train": n_tr, "test": n_te,
             "ratio_max_min": max(n_all.values()) / min(n_all.values()),
             "imbalance_note": f"最多 {CLASSES[np.argmax([n_all[c] for c in CLASSES])]} "
                               f"{max(n_all.values())} vs 最少 {CLASSES[np.argmin([n_all[c] for c in CLASSES])]} "
                               f"{min(n_all.values())}, 比例 {max(n_all.values())/min(n_all.values()):.1f}:1"}

    # ---- 图1: 类别分布条形图 (train+test 堆叠) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(7)
    b1 = ax.bar(x, [n_tr[c] for c in CLASSES], 0.6, label="训练集", color="#4C72B0")
    b2 = ax.bar(x, [n_te[c] for c in CLASSES], 0.6, bottom=[n_tr[c] for c in CLASSES],
                label="测试集", color="#8FB8DE")
    for i, c in enumerate(CLASSES):
        ax.text(i, n_all[c] + 180, f"{n_all[c]}\n({n_all[c]/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(LABELS_ZH)
    ax.set_ylabel("样本数")
    ax.set_title(f"FER-2013 各类表情样本分布 (共 {total} 张, 48×48 灰度)\n"
                 f"不平衡: {CLASSES[np.argmax([n_all[c] for c in CLASSES])]} 最多 "
                 f"{max(n_all.values())} 张 vs {CLASSES[np.argmin([n_all[c] for c in CLASSES])]} 最少 "
                 f"{min(n_all.values())} 张 ({max(n_all.values())/min(n_all.values()):.1f}:1)")
    ax.legend(); ax.set_ylim(0, max(n_all.values()) * 1.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{outdir}/01_class_dist.png", dpi=120); plt.close()

    # ---- 图2: 各类平均脸 (每类像素平均 = 该表情的"典型长相") ----
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for i, c in enumerate(CLASSES):
        mean = tr[c].mean(axis=0)
        ax = axes[i // 4][i % 4]
        ax.imshow(mean, cmap="gray")
        ax.set_title(f"{LABELS_ZH[i]}\n均值亮度 {mean.mean():.1f}", fontsize=10)
        ax.axis("off")
    allmean = np.concatenate([tr[c] for c in CLASSES]).mean(axis=0)
    ax = axes[1][3]
    ax.imshow(allmean, cmap="gray")
    ax.set_title("全部平均脸", fontsize=10); ax.axis("off")
    fig.suptitle("FER-2013 各类表情平均脸 (同类所有图逐像素平均)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{outdir}/02_mean_faces.png", dpi=120); plt.close()

    # ---- 图3: 亮度/对比度统计 ----
    bright = {c: tr[c].mean() for c in CLASSES}
    contrast = {c: tr[c].std() for c in CLASSES}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(7)
    ax1.bar(x, [bright[c] for c in CLASSES], color="#4C72B0")
    ax1.set_xticks(x); ax1.set_xticklabels(LABELS_ZH, fontsize=9)
    ax1.set_title("各类表情平均亮度 (0-255)")
    for i, c in enumerate(CLASSES):
        ax1.text(i, bright[c] + 1, f"{bright[c]:.1f}", ha="center", fontsize=9)
    ax2.bar(x, [contrast[c] for c in CLASSES], color="#55A868")
    ax2.set_xticks(x); ax2.set_xticklabels(LABELS_ZH, fontsize=9)
    ax2.set_title("各类表情像素对比度 (标准差)")
    for i, c in enumerate(CLASSES):
        ax2.text(i, contrast[c] + 0.3, f"{contrast[c]:.1f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{outdir}/03_class_brightness.png", dpi=120); plt.close()

    stats["brightness"] = {c: round(float(bright[c]), 2) for c in CLASSES}
    stats["contrast"] = {c: round(float(contrast[c]), 2) for c in CLASSES}
    json.dump(stats, open(f"{outdir}/fer_eda_stats.json", "w"), ensure_ascii=False, indent=2)
    print("EDA_DONE", file=sys.stderr, flush=True)

if __name__ == "__main__":
    run_eda(sys.argv[1], sys.argv[2])
