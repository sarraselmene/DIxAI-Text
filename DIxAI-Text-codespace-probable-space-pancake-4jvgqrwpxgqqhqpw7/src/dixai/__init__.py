"""
Decision-Information XAI (dixai)
"""

__version__ = "0.1.0"

from .explainer import DecisionInformationExplainer, DecisionInformationExplanation
from .visualization import plot_feature_importance, plot_global_importance, plot_variation_curve
from .metrics import calculate_fidelity, calculate_sparsity, calculate_insertion_deletion_curves, calculate_auc
from .baselines import RISEExplainer, GradCAMPlusPlusExplainer
