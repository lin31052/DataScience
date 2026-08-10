"""Iris 分类训练脚本 (Kaggle CPU 模式)
- 数据: 从 /kaggle/working/kdata/ 读 iris 数据集 (STEP1 已下载解压)
- 模型: LogisticRegression (纯 CPU)
- 产物: out/iris_accuracy.png + out/iris_result.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ===== 读数据路径 (STEP1 写入) =====
with open('/kaggle/working/data_path.txt') as f:
    data_dir = f.read().strip()
log(f'DATA_DIR: {data_dir}')
files = os.listdir(data_dir)
log(f'DATA_FILES: {files}')

# 找 iris csv
csv_path = None
for f in files:
    if f.endswith('.csv'):
        csv_path = os.path.join(data_dir, f)
        break
if csv_path is None:
    log('ERROR: no csv found in data dir')
    sys.exit(1)
log(f'CSV: {csv_path}')

df = pd.read_csv(csv_path)
log(f'SHAPE: {df.shape}')
log(f'COLUMNS: {list(df.columns)}')

# ===== 特征/标签 =====
# uciml/iris 有 Id/SepalLengthCm/SepalWidthCm/PetalLengthCm/PetalWidthCm/Species
label_col = None
for c in df.columns:
    if df[c].dtype == object or 'species' in c.lower() or 'Species' in c:
        label_col = c
        break
if label_col is None:
    log('ERROR: no label column found')
    sys.exit(1)
feature_cols = [c for c in df.columns if c != label_col and c.lower() != 'id']
log(f'LABEL: {label_col}, FEATURES: {feature_cols}')

X = df[feature_cols].values
y = df[label_col].astype(str).values

# ===== 训练/测试划分 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
log(f'TRAIN: {len(X_train)}, TEST: {len(X_test)}')

# ===== 训练 (CPU) =====
log('TRAINING_START')
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train, y_train)
log('TRAINING_DONE')

# ===== 评估 =====
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
log(f'TEST_ACC: {acc:.4f}')
report = classification_report(y_test, y_pred, output_dict=True)

# ===== 保存产物到 out/ =====
outdir = '/kaggle/working/out'
os.makedirs(outdir, exist_ok=True)

# 1. 准确率柱状图
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(['LogisticRegression (CPU)'], [acc], color='#4C72B0', width=0.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy')
ax.set_title(f'Model Test Accuracy: {acc*100:.2f}%')
for bar, v in zip(bars, [acc]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{v:.4f}', ha='center', va='bottom', fontsize=13)
plt.tight_layout()
plt.savefig(f'{outdir}/iris_accuracy.png', dpi=120)
log('ARTIFACT: iris_accuracy.png')

# 2. 结果 JSON
result = {
    'task': 'iris classification (CPU)',
    'model': 'LogisticRegression(lbfgs, max_iter=500)',
    'test_accuracy': float(acc),
    'n_train': len(X_train), 'n_test': len(X_test),
    'report': report
}
with open(f'{outdir}/iris_result.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
log('ARTIFACT: iris_result.json')

log(f'TRAIN_COMPLETE acc={acc:.4f} files={os.listdir(outdir)}')
log('TRAIN_DONE')
