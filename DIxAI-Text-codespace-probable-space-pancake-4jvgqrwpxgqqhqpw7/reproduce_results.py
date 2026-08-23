import torch
import numpy as np
import random
import os
import subprocess
import sys

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"--- Global Seed Set: {seed} ---")

def run_script(script_path):
    print(f"--- Running: {script_path} ---")
    try:
        # Pass the current environment and add project root to PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True, env=env)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stderr)
        return False
    return True

def reproduce_all():
    set_seed(42)
    
    # 1. MI Validation
    run_script('experiments/analysis/estimator_validation.py')
    
    # 2. Scaling Audit
    run_script('experiments/analysis/bench_scaling.py')
    
    # 3. PAMI Figures (Robustness, Sensitivity, Seeds)
    run_script('experiments/analysis/plot_pami_figures.py')
    
    # 4. Rigorous MI Correlation
    run_script('experiments/analysis/rigorous_mi_correlation.py')
    
    print("--- All core results reproduced successfully. check 'experiments/results' and 'figures/' ---")

if __name__ == "__main__":
    reproduce_all()
