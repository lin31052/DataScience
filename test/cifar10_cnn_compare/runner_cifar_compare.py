# DataScience-Kaggle runner: CIFAR-10 三架构对比 (LeNet-5 vs AlexNet vs ResNet18)
# STEP1 kaggle CLI 下载数据集 -> STEP2 clone 代码 -> STEP3 T4 GPU 训练
import subprocess, sys, os

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ===== 配置 =====
GITHUB_USER = "lin31052"
REPO_NAME   = "DataScience"
TRAIN_SCRIPT = "test/cifar10_cnn_compare/train_cifar_compare.py"
DATASET     = "pankrzysiu/cifar10-python"
# ===============

# ===== STEP1: kaggle CLI 下载数据集 =====
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
r = subprocess.run(
    [sys.executable, "-u",
     f"/kaggle/working/DataScience/{TRAIN_SCRIPT}"],
    capture_output=True, text=True)
if r.returncode != 0:
    log('TRAIN_FAILED exit=' + str(r.returncode))
    log('--- stdout tail ---')
    log((r.stdout or '')[-3000:])
    log('--- stderr tail ---')
    log((r.stderr or '')[-3000:])
    sys.exit(1)
log('TRAIN_STDOUT_TAIL: ' + (r.stdout or '')[-500:])
log('ALL_DONE')
