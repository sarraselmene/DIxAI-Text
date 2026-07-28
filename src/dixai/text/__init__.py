"""
DIxAI-Text: extension of DIxAI to NLP / contextual token embeddings (BERT, RoBERTa, ...).
"""

from .masks_text import TokenGumbelSoftmaxMask
from .baselines_text import TextBaselineProvider, BaselineType
from .objective_text import DecisionInformationLossText
from .explainer_text import TextDecisionInformationExplainer, TextExplanation
from .metrics_text import (
    compute_eraser_scores,
    compute_eraser_scores_both_strategies,
    ERASERScores,
)

__all__ = [
    "TokenGumbelSoftmaxMask",
    "TextBaselineProvider",
    "BaselineType",
    "DecisionInformationLossText",
    "TextDecisionInformationExplainer",
    "TextExplanation",
    "compute_eraser_scores",
    "compute_eraser_scores_both_strategies",
    "ERASERScores",
]
