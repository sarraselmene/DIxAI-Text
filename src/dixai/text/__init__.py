"""
DIxAI-Text: extension of DIxAI to NLP / contextual token embeddings (BERT, RoBERTa, ...).

Semaine 3-4 deliverables:
  - masks_text.TokenGumbelSoftmaxMask   -> Verrou 1 (masquage Gumbel-Softmax sur tokens)
  - baselines_text.TextBaselineProvider -> Verrou 2 (baseline B pour le texte)
  - objective_text.DecisionInformationLossText -> Verrou 3 (pénalité de contiguïté séquentielle)
  - explainer_text.TextDecisionInformationExplainer -> assemble les 3 pièces + boucle d'optimisation
"""

from .masks_text import TokenGumbelSoftmaxMask
from .baselines_text import TextBaselineProvider, BaselineType
from .objective_text import DecisionInformationLossText
from .explainer_text import TextDecisionInformationExplainer, TextExplanation
from .metrics_text import compute_eraser_scores, compute_eraser_scores_both_strategies
__all__ = [
    "TokenGumbelSoftmaxMask",
    "TextBaselineProvider",
    "BaselineType",
    "DecisionInformationLossText",
    "TextDecisionInformationExplainer",
    "TextExplanation",
]
