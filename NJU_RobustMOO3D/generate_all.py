from pathlib import Path
import sys

# 工程根目录
ROOT = Path(__file__).resolve().parent

# 确保 src / scripts / robust 在 sys.path 里
for sub in ["src", "scripts", "robust"]:
    p = ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.main_runner import run_all


if __name__ == "__main__":
    run_all(ROOT)
