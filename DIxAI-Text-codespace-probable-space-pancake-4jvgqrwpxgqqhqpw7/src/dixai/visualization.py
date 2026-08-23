
import matplotlib.pyplot as plt
import numpy as np
import torch
from .explainer import DecisionInformationExplanation

def plot_feature_importance(explanation: DecisionInformationExplanation, feature_names=None, title="Decision Information Profile", save_path=None, show=True):
    """
    Plots the feature importance (mask values) as a horizontal bar chart.
    
    Args:
        explanation: The explanation object containing the mask.
        feature_names: List of strings for feature names. If None, uses indices.
        title: Plot title.
        save_path: If provided, saves the plot to this path.
        show: If True, calls plt.show(). Default True.
    """
    mask_values = explanation.mask.numpy().flatten()
    n_features = len(mask_values)
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
        
    # Sort by importance (mask value)
    sorted_idx = np.argsort(mask_values)
    sorted_values = mask_values[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Color map: Low value (discarded) -> Gray, High value (kept) -> Red/Blue
    colors = plt.cm.viridis(sorted_values) # Viridis is safer for visibility
    
    plt.figure(figsize=(10, 6))
    # Add edge color to make sure even thin/light bars are visible
    bars = plt.barh(np.arange(n_features), sorted_values, color=colors, edgecolor='black', alpha=0.8)
    
    plt.yticks(np.arange(n_features), sorted_names)
    plt.xlabel("Decision Information Retention Probability P(Z|X)")
    plt.title(title)
    plt.xlim(0, 1.05) # Force range to show full probability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for i, v in enumerate(sorted_values):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')

    # Add scalar metrics to title or text
    plt.figtext(0.15, 0.02, f"Fidelity: {explanation.fidelity_score:.2f} | Sparsity: {1.0-explanation.info_score:.2f}", fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()

def plot_global_importance(explanations, feature_names=None, title="Global Decision Information Importance", save_path=None, show=True):
    """
    Aggregates explanations to show global feature importance.
    """
    # Stack masks: [N_samples, N_features]
    masks = np.stack([e.mask.numpy().flatten() for e in explanations])
    mean_importance = masks.mean(axis=0)
    std_importance = masks.std(axis=0)
    
    n_features = masks.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
        
    sorted_idx = np.argsort(mean_importance)
    sorted_mean = mean_importance[sorted_idx]
    sorted_std = std_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(n_features), sorted_mean, xerr=sorted_std, color='skyblue', edgecolor='black', capsize=5)
    plt.yticks(np.arange(n_features), sorted_names)
    plt.xlabel("Average Decision Information Retention Probability")
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Global plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

    return plt.gcf()

def plot_variation_curve(sweep_results, title="Fidelity vs Sparsity Trade-off", save_path=None, show=True):
    """
    Plots the trade-off curve from sweep_lambda results.
    """
    lambdas = [r['lambda'] for r in sweep_results]
    fidelities = [r['explanation'].fidelity_score for r in sweep_results]
    sparsities = [1.0 - r['explanation'].info_score for r in sweep_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Lambda (Fidelity Weight)')
    ax1.set_ylabel('Fidelity Score', color=color)
    ax1.plot(lambdas, fidelities, marker='o', color=color, label='Fidelity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Sparsity (1 - Info)', color=color)
    ax2.plot(lambdas, sparsities, marker='s', color=color, linestyle='--', label='Sparsity')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Variation plot saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()

    return fig

def plot_beeswarm(explanations, x_values, x_original=None, feature_names=None, highlight_idx=None, title="Chromatic DIXAI Diagnostic Dashboard", save_path=None, show=True):
    """
    Chromatic V5 version of the Beeswarm plot:
    - High-Contrast Diverging Palette (Red-Blue)
    - Row Banding for visual separation
    - Luminous point aesthetics (white edge glow, higher alpha)
    - Physical point packing logic
    """
    masks = np.stack([e.mask_probs.numpy().flatten() for e in explanations])
    n_samples, n_features = masks.shape
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
        
    mean_importance = masks.mean(axis=0)
    sorted_idx = np.argsort(mean_importance)
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # High-Performance Diverging Colormap (SHAP-standard colors for familiarity)
    cmap = plt.cm.RdBu_r 
    
    for i, f_idx in enumerate(sorted_idx):
        probs = masks[:, f_idx]
        vals = x_values[:, f_idx]
        
        # 1. Row Banding (Zebra layout for better scanning)
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.03, zorder=0)
            
        # 2. Global Distribution Shadow
        ax.barh(i, mean_importance[f_idx], color='black', alpha=0.05, height=0.6, zorder=1)

        # 3. Force-Directed Packing Logic (Improved V5)
        nbins = 45
        bins = np.linspace(-0.01, 1.01, nbins)
        digitized = np.digitize(probs, bins)
        
        y_offsets = np.zeros_like(probs)
        for b in range(1, nbins + 1):
            subset = (digitized == b)
            if subset.any():
                n_points = subset.sum()
                subset_indices = np.where(subset)[0]
                # Order by value to create "color ribbons"
                sorted_in_bin = subset_indices[np.argsort(vals[subset])]
                
                # Stack from center outwards
                offsets = np.linspace(-0.35, 0.35, n_points)
                y_offsets[sorted_in_bin] = offsets

        # 4. Render Chromatic Population
        # Using higher s (size) and white edges for "luminous" look
        sc = ax.scatter(probs, i + y_offsets, c=vals, cmap=cmap, 
                        alpha=0.85, edgecolors='white', linewidth=0.5, s=70, zorder=3)
        
        # 5. Highlight Marker (The Star)
        if highlight_idx is not None:
            hi_prob = probs[highlight_idx]
            hi_y = i + y_offsets[highlight_idx]
            
            # Glowing star
            ax.scatter(hi_prob, hi_y, color='#00FFFF', marker='*', s=350, 
                       edgecolors='black', linewidth=1.5, zorder=20)
            
            if x_original is not None:
                orig_val = x_original[highlight_idx, f_idx]
                label_text = f"{orig_val:.2f}" if isinstance(orig_val, (float, np.float32, np.float64)) else str(orig_val)
                ax.annotate(label_text, (hi_prob, hi_y), xytext=(0, 15), 
                            textcoords='offset points', ha='center', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#00FFFF', alpha=0.9), zorder=21)

        # 6. Sharp Mean Marker
        ax.scatter(mean_importance[f_idx], i, color='#FFD700', marker='|', s=450, linewidth=3, zorder=10)

    # UI/UX Clean-up
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=13, fontweight='black') 
    ax.set_xlabel("Decision Sufficiency P(Z|X) →", fontsize=15, fontweight='bold', labelpad=20)
    ax.set_title(title, fontsize=26, fontweight='black', pad=50)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='|', color='#FFD700', label='Group Average', markersize=18, linewidth=0),
        Line2D([0], [0], marker='*', color='#00FFFF', label='Selected Case', markersize=18, linewidth=0, markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, -0.05), ncol=2, frameon=False, fontsize=12, prop={'weight':'bold'})
    
    # Custom Colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Feature Intensity', fontsize=12, fontweight='bold', labelpad=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontsize=11, fontweight='bold')
    
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chromatic Masterpiece saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig

def plot_lambda_trajectory(sweep_results, feature_names=None, title="Sufficient Feature Hierarchy (Lambda Sweep)", save_path=None, show=True):
    """
    Enhanced Lambda Trajectory with secondary Fidelity axis and bold typography.
    """
    lambdas = [r['lambda'] for r in sweep_results]
    fidelities = [r['explanation'].fidelity_score for r in sweep_results]
    n_features = len(sweep_results[0]['explanation'].mask_probs)
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
        
    fig, ax1 = plt.subplots(figsize=(13, 8))
    
    # Primary axis: Retention probabilities
    for f_idx in range(n_features):
        probs = [r['explanation'].mask_probs[f_idx].item() for r in sweep_results]
        ax1.plot(lambdas, probs, label=feature_names[f_idx], marker='o', markersize=8, linewidth=3)
        
    ax1.set_xscale('log')
    ax1.set_xlabel("Fidelity Weight ($\lambda$)", fontsize=14, fontweight='bold', labelpad=15)
    ax1.set_ylabel("Retention Probability $P(Z|X)$", fontsize=14, fontweight='bold', labelpad=15)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Secondary axis: Global Fidelity
    ax2 = ax1.twinx()
    ax2.plot(lambdas, fidelities, color='black', linestyle='--', alpha=0.5, linewidth=4, label='Global Fidelity')
    ax2.set_ylabel("Model Fidelity", color='black', alpha=0.6, fontsize=14, fontweight='bold', labelpad=15)
    ax2.fill_between(lambdas, fidelities, alpha=0.08, color='black')
    ax2.tick_params(axis='y', labelsize=12)
    
    plt.title(title, fontsize=22, fontweight='black', pad=30)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=12, prop={'weight':'bold'})
    
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig
