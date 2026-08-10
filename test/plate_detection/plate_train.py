#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车牌检测训练 — VOC→YOLO 转换 + YOLOv8n 训练 (GPU) + 结果可视化
数据: andrewmvd/car-plate-detection (433 张, 1 类), VOC xml → YOLO txt
模型: YOLOv8n (nano), T4 GPU, 30 epochs
"""
import sys, json, glob, os, random
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import xml.etree.ElementTree as ET
import pandas as pd

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
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=180)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                       capture_output=True, text=True, timeout=300)
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

OUT = "/kaggle/working"
DATA = "/kaggle/working/kdata/plate"
report = {}

# ---------- 1. VOC → YOLO 转换 ----------
log("STEP: VOC → YOLO 转换...")
xmls = sorted(glob.glob(f"{DATA}/annotations/*.xml"))
random.seed(42)
random.shuffle(xmls)
n_valid = max(1, int(len(xmls) * 0.15))
valid_xmls, train_xmls = xmls[:n_valid], xmls[n_valid:]
log(f"训练 {len(train_xmls)} / 验证 {len(valid_xmls)}")

os.makedirs(f"{OUT}/yolo/images/train", exist_ok=True)
os.makedirs(f"{OUT}/yolo/images/valid", exist_ok=True)
os.makedirs(f"{OUT}/yolo/labels/train", exist_ok=True)
os.makedirs(f"{OUT}/yolo/labels/valid", exist_ok=True)

import shutil
def convert(xml_path, out_img, out_lbl):
    t = ET.parse(xml_path).getroot()
    fname = t.find('filename').text
    w = int(t.find('size').find('width').text)
    h = int(t.find('size').find('height').text)
    src_img = f"{DATA}/images/{fname}"
    shutil.copy(src_img, f"{out_img}/{fname}")
    lines = []
    for obj in t.findall('object'):
        bb = obj.find('bndbox')
        x1, y1 = float(bb.find('xmin').text), float(bb.find('ymin').text)
        x2, y2 = float(bb.find('xmax').text), float(bb.find('ymax').text)
        cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
        bw, bh = (x2-x1)/w, (y2-y1)/h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(f"{out_lbl}/{os.path.splitext(fname)[0]}.txt", "w") as f:
        f.write("\n".join(lines))

for x in train_xmls: convert(x, f"{OUT}/yolo/images/train", f"{OUT}/yolo/labels/train")
for x in valid_xmls: convert(x, f"{OUT}/yolo/images/valid", f"{OUT}/yolo/labels/valid")

with open(f"{OUT}/yolo/data.yaml", "w") as f:
    f.write("path: /kaggle/working/yolo\n")
    f.write("train: images/train\nval: images/valid\n")
    f.write("names:\n  0: license_plate\n")
log("转换完成: " + str(len(os.listdir(f"{OUT}/yolo/images/train"))) + " train / " +
    str(len(os.listdir(f"{OUT}/yolo/images/valid"))) + " valid")

# ---------- 2. 训练 ----------
log("STEP: pip install ultralytics...")
import subprocess
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
                   capture_output=True, text=True)
log('pip rc=' + str(r.returncode))

from ultralytics import YOLO
import torch
log(f"CUDA: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

model = YOLO('yolov8n.pt')
log("STEP: 训练 30 epochs (T4)...")
results = model.train(data=f"{OUT}/yolo/data.yaml", epochs=30, imgsz=640, batch=16,
                      device=0, project=f"{OUT}/plate_run", name="yolov8n", exist_ok=True,
                      verbose=False, seed=42)
log("TRAIN_DONE")

# ---------- 3. 结果指标 ----------
m = model.val(data=f"{OUT}/yolo/data.yaml", device=0, verbose=False)
report['mAP50'] = round(float(m.box.map50), 4)
report['mAP50_95'] = round(float(m.box.map), 4)
report['precision'] = round(float(m.box.mp), 4)
report['recall'] = round(float(m.box.mr), 4)
log(f"mAP50={report['mAP50']} | mAP50-95={report['mAP50_95']} | P={report['precision']} | R={report['recall']}")

# ---------- 4. 结果可视化 ----------
# 4.1 中文 loss 曲线 (读 results.csv 重画)
csv = glob.glob(f"{OUT}/plate_run/yolovn8/results.csv") or glob.glob(f"{OUT}/plate_run/**/results.csv", recursive=True)
if csv:
    rc = pd.read_csv(csv[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(rc['train/box_loss'], label='box loss', color="#4C72B0")
    axes[0].plot(rc['train/cls_loss'], label='cls loss', color="#e74c3c")
    axes[0].plot(rc['train/dfl_loss'], label='dfl loss', color="#55A868")
    axes[0].set_title("训练损失曲线", fontsize=13); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(rc['metrics/precision(B)'], label='precision', color="#4C72B0")
    axes[1].plot(rc['metrics/recall(B)'], label='recall', color="#e74c3c")
    axes[1].plot(rc['metrics/mAP50(B)'], label='mAP50', color="#55A868")
    axes[1].set_title("验证指标曲线", fontsize=13); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/plate_03_metrics_zh.png", dpi=110); plt.close()

# 4.2 验证集预测示例 (6 张) — 匹配所有文件(不依赖扩展名, 兼容 .jpg/.png/.PNG/.jpeg)
valid_dir = f"{OUT}/yolo/images/valid"
valid_imgs = sorted([os.path.join(valid_dir, f) for f in os.listdir(valid_dir)
                     if os.path.isfile(os.path.join(valid_dir, f)) and os.path.getsize(os.path.join(valid_dir, f)) > 1000])[:6]
log(f"预测示例图: {[os.path.basename(p) for p in valid_imgs]}")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, p in zip(axes.ravel(), valid_imgs):
    res = model.predict(p, conf=0.25, verbose=False)[0]
    arr = res.plot()  # BGR
    ax.imshow(arr[..., ::-1])
    ax.set_title(os.path.basename(p), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("验证集检测结果（YOLOv8n, 30 epochs）", fontsize=15)
plt.tight_layout(); plt.savefig(f"{OUT}/plate_04_predictions.png", dpi=110); plt.close()

with open(f"{OUT}/plate_train_result.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
log("ARTIFACTS: " + str(sorted(os.listdir(OUT))))
log('PLATE_TRAIN_DONE')
