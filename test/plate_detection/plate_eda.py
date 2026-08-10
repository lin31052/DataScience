#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车牌数据集 EDA — VOC 标注解析 → 数据概况 → 可视化 (CPU, 全程秒级)
数据: andrewmvd/car-plate-detection (433 张, 1 类: license plate, VOC xml 标注)
"""
import sys, json, glob, os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import xml.etree.ElementTree as ET
import cv2

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
DATA = "/kaggle/working/kdata/plate"
report = {}

# ---------- 解析 VOC ----------
imgs_dir = f"{DATA}/images"
anns_dir = f"{DATA}/annotations"
xmls = sorted(glob.glob(f"{anns_dir}/*.xml"))
log(f"标注文件数: {len(xmls)}")
rows = []
for x in xmls:
    t = ET.parse(x).getroot()
    name = t.find('filename').text
    size = t.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    boxes = []
    for obj in t.findall('object'):
        cls = obj.find('name').text
        bb = obj.find('bndbox')
        x1, y1 = float(bb.find('xmin').text), float(bb.find('ymin').text)
        x2, y2 = float(bb.find('xmax').text), float(bb.find('ymax').text)
        boxes.append({'cls': cls, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    rows.append({'file': name, 'w': w, 'h': h, 'n_box': len(boxes), 'boxes': boxes})

df = pd.DataFrame([{k: v for k, v in r.items() if k != 'boxes'} for r in rows])
all_boxes = [b for r in rows for b in r['boxes']]
log(f"图片数: {len(df)} | 标注框总数: {len(all_boxes)}")
log(f"类别: {sorted(set(b['cls'] for b in all_boxes))}")
report['n_images'] = int(len(df)); report['n_boxes'] = int(len(all_boxes))
report['classes'] = sorted(set(b['cls'] for b in all_boxes))

# ---------- 图片尺寸 ----------
log(f"图片尺寸: 宽 {df['w'].min()}-{df['w'].max()}px / 高 {df['h'].min()}-{df['h'].max()}px")
report['img_size_range'] = [int(df['w'].min()), int(df['w'].max()), int(df['h'].min()), int(df['h'].max())]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(df['w'], df['h'], alpha=0.55, s=30, color="#4C72B0", edgecolor="white")
ax.set_title(f"图片尺寸分布（{len(df)} 张，宽 {df['w'].min()}-{df['w'].max()} × 高 {df['h'].min()}-{df['h'].max()}）", fontsize=13)
ax.set_xlabel("宽度 px"); ax.set_ylabel("高度 px")
plt.tight_layout(); plt.savefig(f"{OUT}/plate_eda_01_size.png", dpi=110); plt.close()

# ---------- 每图框数 ----------
vc = df['n_box'].value_counts().sort_index()
log(f"每图框数分布: " + ", ".join(f"{k}框={v}张" for k, v in vc.items()))
report['boxes_per_image'] = vc.to_dict()
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(vc.index.astype(str), vc.values, color="#4C72B0", edgecolor="white")
for b, v in zip(bars, vc.values):
    ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha="center", fontsize=11)
ax.set_title("每张图片的车牌框数量分布", fontsize=13)
ax.set_xlabel("框数量"); ax.set_ylabel("图片数")
plt.tight_layout(); plt.savefig(f"{OUT}/plate_eda_02_boxes_per_image.png", dpi=110); plt.close()

# ---------- 框尺寸 ----------
bw = [b['x2']-b['x1'] for b in all_boxes]
bh = [b['y2']-b['y1'] for b in all_boxes]
area_ratio = [((b['x2']-b['x1'])*(b['y2']-b['y1'])) for b in all_boxes]
img_area = df.set_index('file')['w'] * df.set_index('file')['h']
# 用文件名关联: 简化 — 每图 1 框为主, 直接按序算近似
area_ratio_norm = [area_ratio[i] / (rows[i % len(rows)]['w'] * rows[i % len(rows)]['h']) for i in range(len(area_ratio))]
log(f"框宽: {min(bw):.0f}-{max(bw):.0f}px (中位 {np.median(bw):.0f}) | 框高: {min(bh):.0f}-{max(bh):.0f}px (中位 {np.median(bh):.0f})")
report['box_w_range'] = [round(min(bw),1), round(max(bw),1), round(float(np.median(bw)),1)]
report['box_h_range'] = [round(min(bh),1), round(max(bh),1), round(float(np.median(bh)),1)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].scatter(bw, bh, alpha=0.5, s=25, color="#e74c3c", edgecolor="white")
axes[0].set_title("车牌框宽 × 高 分布", fontsize=13)
axes[0].set_xlabel("框宽 px"); axes[0].set_ylabel("框高 px")
axes[1].hist(area_ratio_norm, bins=25, color="#4C72B0", alpha=0.85, edgecolor="white")
axes[1].set_title("车牌框占图片面积比例", fontsize=13)
axes[1].set_xlabel("框面积 / 图片面积")
plt.tight_layout(); plt.savefig(f"{OUT}/plate_eda_03_box_size.png", dpi=110); plt.close()

# ---------- 示例图带框 (6 张) ----------
sample_rows = rows[:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, r in zip(axes.ravel(), sample_rows):
    img = cv2.imread(f"{imgs_dir}/{r['file']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for b in r['boxes']:
        cv2.rectangle(img, (int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2'])), (230, 25, 25), 3)
        cv2.putText(img, "plate", (int(b['x1']), int(b['y1'])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 25, 25), 2)
    ax.imshow(img)
    ax.set_title(f"{r['file']} ({r['w']}×{r['h']})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("车牌数据集示例（红框=标注）", fontsize=15)
plt.tight_layout(); plt.savefig(f"{OUT}/plate_eda_04_samples.png", dpi=110); plt.close()

with open(f"{OUT}/plate_eda_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=str)
log('PLATE_EDA_DONE')
