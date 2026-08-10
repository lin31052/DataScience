# DataScience-Kaggle CPU runner: STEP1 kaggle CLI 下载 iris → STEP2 clone 代码 → STEP3 CPU 训练
import subprocess, sys, os

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ===== 使用者必改 =====
GITHUB_USER = "lin31052"
REPO_NAME   = "DataScience"
TRAIN_SCRIPT = "test/iris_cpu/train_iris_sklearn.py"
DATASET     = "uciml/iris"
# =====================

if '<' in GITHUB_USER:
    sys.exit("请在 runner 顶部把 GITHUB_USER 改成你自己的 GitHub 用户名")

# ===== STEP1: kaggle CLI 下载数据集 (CPU 模式, 不烧 GPU 配额) =====
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

# ===== STEP3: CPU 训练 (无 GPU) =====
log('STEP3: running iris sklearn training on CPU...')
subprocess.run(
    [sys.executable, "-u",
     f"/kaggle/working/DataScience/{TRAIN_SCRIPT}"],
    check=True)
log('ALL_DONE')
