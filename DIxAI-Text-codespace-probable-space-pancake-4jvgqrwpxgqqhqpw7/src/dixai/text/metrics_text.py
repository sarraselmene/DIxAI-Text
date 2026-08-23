"""
Verrou 3 — Métriques ERASER pour DIxAI-Text.

Métriques ERASER (DeYoung et al., 2020) :
  - Sufficiency  (Suff) : ne garder QUE les tokens sélectionnés. Plus BAS = mieux.
  - Comprehensiveness (Comp) : RETIRER les tokens sélectionnés. Plus HAUT = mieux.

Formule commune :
    metric = P(y_orig | x_full) - P(y_orig | x_altered)

Deux stratégies de binarisation :
  - threshold=0.5 : standard ERASER.
  - top_k         : garder k% des tokens de contenu (k configurable).

Les tokens spéciaux ([CLS]/[SEP]/[PAD]) sont exclus du calcul.
"""

import torch
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass


@dataclass
class ERASERScores:
    sufficiency: float
    comprehensiveness: float
    n_tokens_kept: int
    n_tokens_total: int
    strategy: str


def _binarize_mask_threshold(
    mask_probs: torch.Tensor,
    content_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Binarise mask_probs par seuillage.
    mask_probs: (T,), content_mask: (T,) -> binary (T,) restreint au contenu.
    """
    binary = (mask_probs > threshold).float()
    return binary * content_mask


def _binarize_mask_topk(
    mask_probs: torch.Tensor,
    content_mask: torch.Tensor,
    k_fraction: float = 0.2,
) -> torch.Tensor:
    """
    Garde les k_fraction% tokens de contenu avec les scores les plus élevés.
    Au moins 1 token gardé si le contenu n'est pas vide.
    """
    content_indices = torch.nonzero(content_mask > 0, as_tuple=False).squeeze(-1)
    n_content = content_indices.numel()
    binary = torch.zeros_like(mask_probs)

    if n_content == 0:
        return binary

    k = max(1, int(round(k_fraction * n_content)))
    content_scores = mask_probs[content_indices]
    _, top_local_idx = torch.topk(content_scores, k=min(k, n_content))
    selected_global_idx = content_indices[top_local_idx]
    binary[selected_global_idx] = 1.0
    return binary


def _predict_proba_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Retourne softmax(logits) depuis embeddings."""
    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out
        return torch.softmax(logits, dim=-1)


def compute_eraser_scores(
    model,
    tokenizer,
    text: str,
    mask_probs: torch.Tensor,
    baseline_vec: torch.Tensor,
    device: str = "cpu",
    strategy: str = "threshold",
    threshold: float = 0.5,
    k_fraction: float = 0.2,
) -> ERASERScores:
    """
    Calcule sufficiency et comprehensiveness pour UNE explication.

    Args:
        model       : modèle HF ForSequenceClassification.
        tokenizer   : tokenizer HF correspondant.
        text        : phrase originale (même que celle passée à explain()).
        mask_probs  : (T,) probabilités continues dans [0,1].
        baseline_vec: (D,) vecteur baseline (même baseline_type que l'explainer).
        device      : "cpu" ou "cuda".
        strategy    : "threshold" ou "top_k".
        threshold   : seuil si strategy="threshold".
        k_fraction  : fraction si strategy="top_k".

    Returns:
        ERASERScores(sufficiency, comprehensiveness, n_tokens_kept, n_tokens_total, strategy)
    """
    dev = torch.device(device)
    embedding_layer = model.get_input_embeddings()

    # --- Tokenisation ---
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc["attention_mask"].to(dev)
    T = input_ids.shape[1]

    # --- Vérification shape mask_probs ---
    mask_probs = mask_probs.flatten().to(dev)
    if mask_probs.shape[0] != T:
        raise RuntimeError(
            f"mask_probs length mismatch: {mask_probs.shape[0]} vs T={T}. "
            f"Assurez-vous que text est identique à celui passé à explain()."
        )

    # --- Content mask ---
    special_ids_list = tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )
    special_tokens_mask = torch.tensor(
        special_ids_list, device=dev, dtype=attention_mask.dtype
    )
    content_mask = attention_mask[0] * (1 - special_tokens_mask)  # (T,)

    # --- Binarisation ---
    if strategy == "threshold":
        selected = _binarize_mask_threshold(mask_probs, content_mask, threshold)
        strategy_label = f"threshold(thr={threshold})"
    elif strategy == "top_k":
        selected = _binarize_mask_topk(mask_probs, content_mask, k_fraction)
        strategy_label = f"top_k(k={k_fraction})"
    else:
        raise ValueError(f"strategy inconnue: {strategy}. Choisir 'threshold' ou 'top_k'.")

    with torch.no_grad():
        x_embeds = embedding_layer(input_ids)  # (1, T, D)
        baseline_vec = baseline_vec.to(dev)

        # Vérification dimension baseline
        D = x_embeds.shape[-1]
        if baseline_vec.shape[0] != D:
            raise RuntimeError(
                f"baseline_vec dim mismatch: {baseline_vec.shape[0]} vs D={D}."
            )

        baseline_expanded = baseline_vec.unsqueeze(0).unsqueeze(0).expand(1, T, -1)

        # --- Prédiction originale ---
        p_orig = _predict_proba_from_embeds(model, x_embeds, attention_mask)
        y_orig_class = p_orig.argmax(dim=-1)

        # --- Sufficiency : garder SEULEMENT les tokens sélectionnés ---
        # Tokens spéciaux toujours gardés (structurels)
        keep_for_suff = (
            selected + special_tokens_mask.float()
        ).clamp(max=1.0).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        z_suff = keep_for_suff * x_embeds + (1 - keep_for_suff) * baseline_expanded
        p_suff = _predict_proba_from_embeds(model, z_suff, attention_mask)

        # --- Comprehensiveness : RETIRER les tokens sélectionnés ---
        keep_for_comp = (
            (1 - selected) * (1 - special_tokens_mask.float())
            + special_tokens_mask.float()
        ).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        z_comp = keep_for_comp * x_embeds + (1 - keep_for_comp) * baseline_expanded
        p_comp = _predict_proba_from_embeds(model, z_comp, attention_mask)

        # --- Scores ERASER ---
        p_full = p_orig[0, y_orig_class].item()
        p_s    = p_suff[0, y_orig_class].item()
        p_c    = p_comp[0, y_orig_class].item()

        sufficiency      = p_full - p_s   # bas = mieux (rationale suffit)
        comprehensiveness = p_full - p_c  # haut = mieux (rationale nécessaire)

    return ERASERScores(
        sufficiency=sufficiency,
        comprehensiveness=comprehensiveness,
        n_tokens_kept=int(selected.sum().item()),
        n_tokens_total=int(content_mask.sum().item()),
        strategy=strategy_label,
    )


def compute_eraser_scores_both_strategies(
    model,
    tokenizer,
    text: str,
    mask_probs: torch.Tensor,
    baseline_vec: torch.Tensor,
    device: str = "cpu",
    threshold: float = 0.5,
    k_fraction: float = 0.2,
) -> Dict[str, ERASERScores]:
    """
    Calcule les scores ERASER avec threshold=0.5 ET top_k, pour comparaison.

    Returns:
        dict avec clés "threshold" et "top_k", valeurs ERASERScores.
    """
    return {
        "threshold": compute_eraser_scores(
            model, tokenizer, text, mask_probs, baseline_vec,
            device=device, strategy="threshold", threshold=threshold,
        ),
        "top_k": compute_eraser_scores(
            model, tokenizer, text, mask_probs, baseline_vec,
            device=device, strategy="top_k", k_fraction=k_fraction,
        ),
    }
