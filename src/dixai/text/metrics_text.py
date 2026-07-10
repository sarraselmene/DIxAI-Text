"""
Verrou 3 — Métriques ERASER pour DIxAI-Text.

Implémente les deux métriques standard du benchmark ERASER
(DeYoung et al., 2020, "ERASER: A Benchmark to Evaluate Rationalized
NLP Models") :

  - Sufficiency (Suff)      : ne garder QUE les tokens sélectionnés (le
                               reste -> baseline). La prédiction doit rester
                               proche de l'originale. Plus BAS = mieux.
  - Comprehensiveness (Comp): RETIRER les tokens sélectionnés (le reste ->
                               baseline). La prédiction doit changer
                               fortement (preuve que les tokens choisis
                               étaient vraiment importants). Plus HAUT = mieux.

Les deux métriques utilisent la même formule de base :

    metric = P(y_orig | x_full) - P(y_orig | x_altered)

où x_altered dépend de la métrique :
    - Sufficiency      : x_altered = seulement les tokens sélectionnés
    - Comprehensiveness: x_altered = tout SAUF les tokens sélectionnés

Deux stratégies de binarisation du mask continu (mask_probs dans [0,1])
sont supportées, car le choix du seuil affecte significativement les
scores et c'est un point à documenter/ablationner dans le rapport :

  - threshold=0.5   : binarisation standard ERASER (mask_probs > 0.5)
  - top_k           : garder les k% de tokens de contenu avec les
                       mask_probs les plus élevés (k configurable),
                       utile pour comparer à parcimonie fixée entre
                       plusieurs méthodes (DIxAI-Text vs LIME vs SHAP...)

Dans les deux cas, les tokens spéciaux ([CLS]/[SEP]/[PAD]) sont exclus
du calcul de comprehensiveness/sufficiency : ce ne sont pas des tokens
de "contenu" candidats, et les compter fausserait la comparaison avec
les baselines externes qui ne les considèrent pas non plus.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ERASERScores:
    sufficiency: float
    comprehensiveness: float
    n_tokens_kept: int
    n_tokens_total: int
    strategy: str  # "threshold" ou "top_k"


def _binarize_mask_threshold(
    mask_probs: torch.Tensor,
    content_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """mask_probs: (T,), content_mask: (T,) -> binary (T,) restreint au contenu."""
    binary = (mask_probs > threshold).float()
    return binary * content_mask


def _binarize_mask_topk(
    mask_probs: torch.Tensor,
    content_mask: torch.Tensor,
    k_fraction: float = 0.2,
) -> torch.Tensor:
    """
    Garde les k_fraction (ex: 0.2 = 20%) tokens de CONTENU (hors spéciaux/
    padding) avec les scores les plus élevés. Au moins 1 token gardé si
    le contenu n'est pas vide.
    """
    content_indices = torch.nonzero(content_mask > 0, as_tuple=False).squeeze(-1)
    n_content = content_indices.numel()
    binary = torch.zeros_like(mask_probs)

    if n_content == 0:
        return binary

    k = max(1, int(round(k_fraction * n_content)))
    content_scores = mask_probs[content_indices]
    top_vals, top_local_idx = torch.topk(content_scores, k=min(k, n_content))
    selected_global_idx = content_indices[top_local_idx]
    binary[selected_global_idx] = 1.0
    return binary


def _predict_proba(
    model,
    embedding_layer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Retourne softmax(logits) pour l'input donné (par ids, pas embeds)."""
    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = out.logits if hasattr(out, "logits") else out
        return torch.softmax(logits, dim=-1)


def _predict_proba_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
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
    Calcule sufficiency et comprehensiveness pour UNE explication déjà
    produite par TextDecisionInformationExplainer.

    Args:
        model: modèle HF ForSequenceClassification (même que l'explainer).
        tokenizer: tokenizer HF correspondant.
        text: phrase originale (doit être la même que celle passée à explain()).
        mask_probs: (T,) mask_probs retourné par explanation.mask_probs
            (probabilités continues dans [0,1], PAS encore binarisées).
        baseline_vec: (D,) vecteur de baseline, ex.
            TextBaselineProvider.get_baseline_vector() (même baseline_type
            que celle utilisée pour produire l'explication, pour rester
            cohérent).
        strategy: "threshold" ou "top_k".
        threshold: utilisé si strategy="threshold".
        k_fraction: utilisé si strategy="top_k" (fraction de tokens de
            contenu à garder, ex 0.2 = 20%).

    Returns:
        ERASERScores(sufficiency, comprehensiveness, n_tokens_kept, ...)
    """
    dev = torch.device(device)
    embedding_layer = model.get_input_embeddings()

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc["attention_mask"].to(dev)
    T = input_ids.shape[1]

    special_ids_list = tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )
    special_tokens_mask = torch.tensor(special_ids_list, device=dev, dtype=attention_mask.dtype)
    content_mask = attention_mask[0] * (1 - special_tokens_mask)

    mask_probs = mask_probs.to(dev)
    assert mask_probs.shape[0] == T, "mask_probs doit correspondre à la même tokenisation que `text`"

    if strategy == "threshold":
        selected = _binarize_mask_threshold(mask_probs, content_mask, threshold)
    elif strategy == "top_k":
        selected = _binarize_mask_topk(mask_probs, content_mask, k_fraction)
    else:
        raise ValueError(f"strategy inconnue: {strategy}")

    with torch.no_grad():
        x_embeds = embedding_layer(input_ids)  # (1, T, D)
        baseline_vec = baseline_vec.to(dev)
        baseline_expanded = baseline_vec.unsqueeze(0).unsqueeze(0).expand(1, T, -1)  # (1, T, D)

        # Prédiction originale (texte complet)
        p_orig = _predict_proba_from_embeds(model, x_embeds, attention_mask)  # (1, C)
        y_orig_class = p_orig.argmax(dim=-1)  # classe prédite originale

        # --- Sufficiency : ne garder QUE les tokens sélectionnés ---
        # Les tokens spéciaux restent toujours (nécessaires structurellement).
        keep_for_suff = (selected + special_tokens_mask.float()).clamp(max=1.0)
        keep_for_suff = keep_for_suff.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        z_suff = keep_for_suff * x_embeds + (1 - keep_for_suff) * baseline_expanded
        p_suff = _predict_proba_from_embeds(model, z_suff, attention_mask)

        # --- Comprehensiveness : RETIRER les tokens sélectionnés ---
        # (le reste, hors spéciaux, est gardé)
        keep_for_comp = ((1 - selected) * (1 - special_tokens_mask.float()) + special_tokens_mask.float())
        keep_for_comp = keep_for_comp.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        z_comp = keep_for_comp * x_embeds + (1 - keep_for_comp) * baseline_expanded
        p_comp = _predict_proba_from_embeds(model, z_comp, attention_mask)

        # Score = proba de la classe originale, sur x_full vs x_altered
        p_orig_class_full = p_orig[0, y_orig_class].item()
        p_orig_class_suff = p_suff[0, y_orig_class].item()
        p_orig_class_comp = p_comp[0, y_orig_class].item()

        sufficiency = p_orig_class_full - p_orig_class_suff
        comprehensiveness = p_orig_class_full - p_orig_class_comp

    return ERASERScores(
        sufficiency=sufficiency,
        comprehensiveness=comprehensiveness,
        n_tokens_kept=int(selected.sum().item()),
        n_tokens_total=int(content_mask.sum().item()),
        strategy=f"{strategy}" + (f"(thr={threshold})" if strategy == "threshold" else f"(k={k_fraction})"),
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
    """Calcule les scores ERASER avec threshold=0.5 ET top_k, pour comparaison."""
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