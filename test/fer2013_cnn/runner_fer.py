# FER-2013 表情识别 runner — Kaggle notebook 入口
# 流程: clone 仓库 → kaggle CLI 下载数据 → EDA → CNN 训练 → 产物落盘
import subprocess, sys, os, zipfile

GITHUB_USER = "lin31052"
REPO_URL = f"https://github.com/{GITHUB_USER}/DataScience.git"
TASK_DIR = "test/fer2013_cnn"
DATA_ROOT = "/kaggle/working/data"
OUTDIR = "/kaggle/working/out"

def log(msg):
    print(msg, file=sys.stderr, flush=True)

log("STEP0: clone repo ...")
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/kaggle/working/DataScience"], check=True)

log("STEP1: download fer2013 via kaggle CLI ...")
subprocess.run(["kaggle", "datasets", "download", "-d", "msambare/fer2013",
                "-p", "/kaggle/working/data"], check=True)
with zipfile.ZipFile("/kaggle/working/data/fer2013.zip") as z:
    z.extractall(DATA_ROOT)
log(f"STEP1_DONE: {sorted(os.listdir(DATA_ROOT))}")

sys.path.insert(0, f"/kaggle/working/DataScience/{TASK_DIR}")
import fer_eda
log("STEP2: EDA ...")
fer_eda.run_eda(DATA_ROOT, OUTDIR)
log("STEP2_DONE")

log("STEP3: train CNN on GPU ...")
import fer_train
fer_train.run_train(DATA_ROOT, OUTDIR)
log("ALL_DONE")
