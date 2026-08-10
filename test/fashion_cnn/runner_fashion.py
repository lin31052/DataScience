# DataScience-Kaggle GPU runner: STEP1 kaggle CLI 下载 Fashion-MNIST → STEP2 clone 代码 → STEP3 T4 训练
import subprocess, sys, os

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

GITHUB_USER = "lin31052"
REPO_NAME   = "DataScience"
TRAIN_SCRIPT = "test/fashion_cnn/train_fashion_cnn.py"
DATASET     = "zalando-research/fashionmnist"

# ===== STEP1: kaggle CLI 下载数据集 (容器自带 KAGGLE_API_V1_TOKEN 认证) =====
log(f'STEP1: downloading dataset {DATASET} via kaggle CLI...')
r = subprocess.run(['kaggle', 'datasets', 'download', '-d', DATASET, '-p', '/kaggle/working/kdata'],
                   capture_output=True, text=True)
if r.returncode != 0:
    log('DOWNLOAD_FAILED: ' + (r.stderr or r.stdout)[-300:])
    sys.exit(1)
log('STEP1_DOWNLOAD_OK: ' + str(os.listdir('/kaggle/working/kdata')))
for z in os.listdir('/kaggle/working/kdata'):
    if z.endswith('.zip'):
        subprocess.run(['unzip', '-o', '-q', f'/kaggle/working/kdata/{z}', '-d', '/kaggle/working/kdata'], check=True)
        os.remove(f'/kaggle/working/kdata/{z}')
log('STEP1_UNZIP_OK: ' + str(os.listdir('/kaggle/working/kdata')))
with open('/kaggle/working/data_path.txt', 'w') as f:
    f.write('/kaggle/working/kdata')
log('STEP1_DONE')

# ===== STEP2: clone 代码仓库 =====
log('STEP2: cloning code repo...')
subprocess.run(
    ["git", "clone", "--depth", "1",
     f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git",
     "/kaggle/working/DataScience"],
    check=True)
log('STEP2_DONE')

# ===== STEP3: T4 GPU 训练 =====
log('STEP3: training on GPU...')
subprocess.run(
    [sys.executable, "-u",
     f"/kaggle/working/DataScience/{TRAIN_SCRIPT}"],
    check=True)
log('ALL_DONE')
