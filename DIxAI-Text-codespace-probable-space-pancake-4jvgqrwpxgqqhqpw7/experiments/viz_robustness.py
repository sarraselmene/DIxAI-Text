
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_robustness():
    df = pd.read_csv("experiments/results/robustness_stats.csv")
    
    # Filter out baseline (level 0) if present, though processed script filters it
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    # Create combined category
    df['Condition'] = df['perturbation'] + " " + df['level'].astype(str)
    
    sns.boxplot(x='Condition', y='iou', data=df, palette="viridis")
    plt.title("DIxAI Robustness: Mask Consistency under Perturbation")
    plt.ylabel("Intersection over Union (IoU) with Baseline Mask")
    plt.xlabel("Perturbation Type & Intensity")
    plt.ylim(0, 1)
    
    # Add stability reference
    plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Stability (Seed=42)')
    plt.legend()
    
    output_path = "experiments/results/robustness_boxplot.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved robustness plot to {output_path}")

if __name__ == "__main__":
    plot_robustness()
