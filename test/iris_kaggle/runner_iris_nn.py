# DataScience-Kaggle v1.4.0 demo runner: STEP1 kaggle CLI 下载数据集 → STEP2 clone 代码 → STEP3 T4 训练
# ⚠️ 首次使用必改: GITHUB_USER 改成你自己的 GitHub 用户名; 仓库公开; demo 代码已 push 到 test/iris_kaggle/
import subprocess, sys, os

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

# ===== 使用者必改(占位符) =====
GITHUB_USER = "lin31052"     # 例: "zhangsan", 不要带 https://
REPO_NAME   = "DataScience"            # 仓库名, 可改成你自己的
TRAIN_SCRIPT = "test/iris_kaggle/train_iris_nn.py"  # demo 训练脚本在仓库里的相对路径
DATASET     = "uciml/iris"             # Kaggle 数据集 (owner/dataset), 可换成其他
# =============================

if '<' in GITHUB_USER:
    sys.exit("请在 runner 顶部把 GITHUB_USER 改成你自己的 GitHub 用户名")

# ===== STEP1: kaggle CLI 下载数据集 (容器自带 KAGGLE_API_V1_TOKEN 认证, 实测可行) =====
log(f'STEP1: downloading dataset {DATASET} via kaggle CLI...')
r = subprocess.run(['kaggle', 'datasets', 'download', '-d', DATASET, '-p', '/kaggle/working/kdata'],
                   capture_output=True, text=True)
if r.returncode != 0:
    log('DOWNLOAD_FAILED: ' + (r.stderr or r.stdout)[-300:])
    log('检查: ①数据集是否存在(Kaggle 搜索确认 owner/name) ②容器联网(--internet)')
    sys.exit(1)
log('STEP1_DOWNLOAD_OK: ' + str(os.listdir('/kaggle/working/kdata')))
# 解压所有 zip (kaggle CLI 下载的是 zip)
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
