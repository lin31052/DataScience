# GPU 冒烟测试 runner: clone 代码 → 跑 FashionMNIST MLP 2 epochs → 验证 T4 可用
# 产物落在 /kaggle/working/ 根目录: model.pth / loss_curve.png / result.json
import subprocess, sys

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

GITHUB_USER = "lin31052"
REPO_NAME = "DataScience"
TRAIN_SCRIPT = "test/gpu_smoke/train_gpu_smoke.py"

log('STEP1: cloning code repo...')
subprocess.run(
    ["git", "clone", "--depth", "1",
     f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git",
     "/kaggle/working/DataScience"],
    check=True)
log('STEP1_DONE')

log('STEP2: running GPU smoke test...')
subprocess.run(
    [sys.executable, "-u", f"/kaggle/working/DataScience/{TRAIN_SCRIPT}"],
    check=True)
log('ALL_DONE')
