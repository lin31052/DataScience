# DataScience-Kaggle v1.5.4 GPU runner: STEP1 下载车牌字符数据 → STEP2 clone → STEP3 EDA(CPU) → STEP4 训练(GPU)
import subprocess, sys, os

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

GITHUB_USER = "lin31052"
REPO_NAME   = "DataScience"
DATASET     = "aladdinss/license-plate-digits-classification-dataset"

if '<' in GITHUB_USER:
    sys.exit("请把 GITHUB_USER 改成你自己的 GitHub 用户名")

# ===== STEP1: 下载数据集 =====
log(f'STEP1: downloading {DATASET} ...')
d = '/kaggle/working/kdata/plate_char'
os.makedirs(d, exist_ok=True)
r = subprocess.run(['kaggle', 'datasets', 'download', '-d', DATASET, '-p', d],
                   capture_output=True, text=True)
if r.returncode != 0:
    log('DOWNLOAD_FAILED: ' + (r.stderr or r.stdout)[-300:]); sys.exit(1)
log('STEP1_DOWNLOAD_OK: ' + str(os.listdir(d)))
for z in os.listdir(d):
    if z.endswith('.zip'):
        subprocess.run(['unzip', '-o', '-q', f'{d}/{z}', '-d', d], check=True)
        os.remove(f'{d}/{z}')
log('STEP1_UNZIP_OK: ' + str([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]))
log('STEP1_DONE')

# ===== STEP2: clone 代码仓库 =====
log('STEP2: cloning code repo...')
subprocess.run(["git", "clone", "--depth", "1",
                f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git",
                "/kaggle/working/DataScience"], check=True)
log('STEP2_DONE')

# ===== STEP3: 数据 EDA (CPU) =====
log('STEP3: running plate char EDA on CPU...')
subprocess.run([sys.executable, "-u",
                "/kaggle/working/DataScience/test/plate_char/plate_char_eda.py"], check=True)
log('STEP3_DONE')

# ===== STEP4: 训练 (GPU) =====
log('STEP4: training CNN on GPU...')
subprocess.run([sys.executable, "-u",
                "/kaggle/working/DataScience/test/plate_char/plate_char_train.py"], check=True)
log('ALL_DONE')
