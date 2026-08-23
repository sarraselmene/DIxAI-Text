import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_pami_figures():
    # Style
    sns.set_theme(style="whitegrid")
    os.makedirs('figures', exist_ok=True)
    
    # --- 1. Rotation Robustness ---
    rot_path = 'experiments/results/rotation_robustness.csv'
    if os.path.exists(rot_path):
        try:
            df_rot = pd.read_csv(rot_path)
            plt.figure(figsize=(7, 5))
            sns.lineplot(data=df_rot, x='Angle', y='IoU', marker='o', markersize=10, linewidth=3, color='#c0392b')
            plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal')
            
            plt.title('DIxAI Explanation Stability under Rotation', fontsize=15, fontweight='bold', pad=15)
            plt.xlabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
            plt.ylabel('Mask IoU (w/ 0° baseline)', fontsize=12, fontweight='bold')
            
            # Show full stability range (including model-flip sensitivity)
            plt.ylim(0.0, 1.1) 
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('figures/rotation_robustness.png', dpi=300)
            print("Standard Rotation Plot Saved.")
        except Exception as e:
            print(f"Error plotting rotation: {e}")
    else:
        print(f"Warning: {rot_path} not found.")

    # --- 2. Baseline Sensitivity ---
    base_path = 'experiments/results/baseline_sensitivity.csv'
    if os.path.exists(base_path):
        try:
            df_base = pd.read_csv(base_path)
            # Map Density to Sparsity for plot
            if 'Density' in df_base.columns:
                df_base['Sparsity'] = 1.0 - df_base['Density']
            
            if 'Sample' in df_base.columns:
                plt.figure(figsize=(9, 6))
                df_melt = df_base.melt(id_vars=['Sample', 'Baseline'], value_vars=['Fidelity', 'Sparsity'], var_name='Metric', value_name='Score')
                ax = sns.barplot(data=df_melt, x='Baseline', y='Score', hue='Metric', palette='viridis', capsize=.1, edgecolor='0.2')
                
                plt.title('DIxAI Sensitivity to Baseline Choice (ImageNet)', fontsize=15, fontweight='bold', pad=20)
                plt.xlabel('Baseline Initialization', fontsize=12, fontweight='bold')
                plt.ylabel('Score', fontsize=12, fontweight='bold')
                plt.ylim(0, 1.1)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                
                # Add mean labels
                for p in ax.patches:
                    if p.get_height() > 0:
                        ax.annotate(format(p.get_height(), '.2f'), 
                                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha='center', va='center', 
                                   xytext=(0, 9), 
                                   textcoords='offset points',
                                   fontsize=9, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig('figures/baseline_sensitivity.png', dpi=300, bbox_inches='tight')
                print("Refined Baseline Sensitivity Plot Saved.")
        except Exception as e:
            print(f"Error plotting baseline: {e}")
    else:
        print(f"Warning: {base_path} not found.")

    # --- 3. Seed Stability (New) ---
    seed_path = 'experiments/results/robustness_stats.csv'
    if os.path.exists(seed_path):
        try:
            df_seed = pd.read_csv(seed_path)
            # Boxplot of Fidelity/Sparsity across seeds for DIxAI vs others
            plt.figure(figsize=(8, 5))
            # Filter methods if needed (DIxAI, L2X, IG)
            df_melt = df_seed.melt(id_vars=['Method'], value_vars=['Fidelity', 'Sparsity'], var_name='Metric', value_name='Score')
            
            sns.boxplot(data=df_melt, x='Method', y='Score', hue='Metric', palette=['#3498db', '#e74c3c'])
            plt.title('Stability Across Random Seeds (N=10)', fontsize=14, fontweight='bold')
            plt.xlabel('Method', fontsize=12)
            plt.ylabel('Score Distribution', fontsize=12)
            plt.ylim(0, 1.1)
            plt.legend(title='')
            plt.tight_layout()
            plt.savefig('figures/seed_stability.png', dpi=300)
            print("Seed Stability Plot Saved.")
        except Exception as e:
            print(f"Error plotting seed stability: {e}")
    else:
        print(f"Warning: {seed_path} not found.")

if __name__ == "__main__":
    plot_pami_figures()
