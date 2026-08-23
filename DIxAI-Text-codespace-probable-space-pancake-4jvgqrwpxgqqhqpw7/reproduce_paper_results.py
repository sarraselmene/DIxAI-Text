import argparse
import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n--- {description} ---")
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✔ {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"✘ Error running {description}: {e}")
        sys.exit(1)

def reproduce_table_1():
    """Reproduce Tabular Baselines (Adult/Credit)"""
    run_command("python experiments/benchmark/bench_tabular_baselines.py", "Table 1: Tabular Baselines")

def reproduce_table_2():
    """Reproduce Higgs Boson Results"""
    run_command("python experiments/benchmark/bench_tabular.py --dataset higgs", "Table 2: Higgs Boson Benchmark")

def reproduce_table_3():
    """Reproduce Faithfulness (AUC)"""
    run_command("python experiments/benchmark/exp2_faithfulness.py", "Table 3: Faithfulness Evaluation")

def reproduce_table_5():
    """Reproduce ImageNet Results"""
    run_command("python experiments/vision/bench_imagenet.py", "Table 5: ImageNet Benchmarks")

def reproduce_latency():
    """Reproduce Latency/Throughput Table"""
    run_command("python experiments/analysis/explainer_profile.py", "Table 6: Computational Characterization")

def reproduce_bias_sweep():
    """Reproduce MI Bias Sweep (Figure 3)"""
    run_command("python experiments/analysis/estimator_validation.py", "Figure 3: MI Bias Sweep")

def train_all():
    """Train Amortized Explainer"""
    print("Warning: Training requires ImageNet data and significant GPU time.")
    run_command("python experiments/main_train_amortized.py --dry-run", "Training Dry-Run Verification")

def main():
    parser = argparse.ArgumentParser(description="Reproduce DIxAI Paper Results")
    parser.add_argument("--all", action="store_true", help="Run all experiments (long runtime!)")
    parser.add_argument("--table1", action="store_true", help="Reproduce Table 1 (Tabular Baselines)")
    parser.add_argument("--table2", action="store_true", help="Reproduce Table 2 (Higgs)")
    parser.add_argument("--table3", action="store_true", help="Reproduce Table 3 (Faithfulness)")
    parser.add_argument("--table5", action="store_true", help="Reproduce Table 5 (ImageNet)")
    parser.add_argument("--latency", action="store_true", help="Reproduce Latency Table")
    parser.add_argument("--bias", action="store_true", help="Reproduce MI Bias Sweep")
    parser.add_argument("--train", action="store_true", help="Verify Training Script")
    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    print("==================================================")
    print("       DIxAI PAMI Reproducibility Suite           ")
    print("==================================================")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split(' ')[0]}")
    
    if args.all or args.table1: reproduce_table_1()
    if args.all or args.table2: reproduce_table_2()
    if args.all or args.table3: reproduce_table_3()
    if args.all or args.table5: reproduce_table_5()
    if args.all or args.latency: reproduce_latency()
    if args.all or args.bias: reproduce_bias_sweep()
    if args.train: train_all()

if __name__ == "__main__":
    main()
