# CV demo runner: clone 仓库 → 跑 test/cv_demo_cifar10/train_cv_demo.py（CIFAR-10 CNN, T4）
import subprocess, sys

def log(msg):
    print(msg, flush=True)
    print('PROGRESS: ' + msg, file=sys.stderr, flush=True)

GITHUB_USER = "lin31052"
REPO_NAME = "DataScience"
TRAIN_SCRIPT = "test/cv_demo_cifar10/train_cv_demo.py"

log('STEP1: cloning code repo...')
subprocess.run(
    ["git", "clone", "--depth", "1",
     f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git",
     "/kaggle/working/DataScience"],
    check=True)
log('STEP1_DONE')

log('STEP2: running CV demo (CIFAR-10 + SmallCNN)...')
subprocess.run(
    [sys.executable, "-u", f"/kaggle/working/DataScience/{TRAIN_SCRIPT}"],
    check=True)
log('ALL_DONE')
