# DataScience-Kaggle v1.5.3 CPU runner: STEP1 下载 heart+stroke 两数据集 → STEP2 clone → STEP3 顺序跑两个分析
import subprocess, sys, os

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

GITHUB_USER = "lin31052"
REPO_NAME   = "DataScience"
EDA_SCRIPTS = ["test/heart_stroke/heart_eda.py", "test/heart_stroke/stroke_eda.py"]
DATASETS    = [("fedesoriano/heart-failure-prediction", "heart"),
               ("fedesoriano/stroke-prediction-dataset", "stroke")]

if '<' in GITHUB_USER:
    sys.exit("请把 GITHUB_USER 改成你自己的 GitHub 用户名")

# ===== STEP1: kaggle CLI 下载两个数据集 =====
for ds, sub in DATASETS:
    log(f'STEP1: downloading {ds} ...')
    d = f'/kaggle/working/kdata/{sub}'
    os.makedirs(d, exist_ok=True)
    r = subprocess.run(['kaggle', 'datasets', 'download', '-d', ds, '-p', d],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log('DOWNLOAD_FAILED: ' + (r.stderr or r.stdout)[-300:]); sys.exit(1)
    log(f'STEP1_DOWNLOAD_OK [{sub}]: ' + str(os.listdir(d)))
    for z in os.listdir(d):
        if z.endswith('.zip'):
            subprocess.run(['unzip', '-o', '-q', f'{d}/{z}', '-d', d], check=True)
            os.remove(f'{d}/{z}')
    log(f'STEP1_UNZIP_OK [{sub}]: ' + str(os.listdir(d)))
log('STEP1_DONE')

# ===== STEP2: clone 代码仓库 =====
log('STEP2: cloning code repo...')
subprocess.run(
    ["git", "clone", "--depth", "1",
     f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git",
     "/kaggle/working/DataScience"],
    check=True)
log('STEP2_DONE')

# ===== STEP3: CPU 分析 (无 GPU) =====
for s in EDA_SCRIPTS:
    log(f'STEP3: running {s} ...')
    subprocess.run([sys.executable, "-u", f"/kaggle/working/DataScience/{s}"], check=True)
log('ALL_DONE')
