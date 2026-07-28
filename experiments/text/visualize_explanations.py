"""
visualize_explanations.py — Visualisation qualitative des explications DIxAI-Text.

Produit :
  1. Visualisation HTML : tokens colorés selon mask_probs (rouge = sélectionné).
  2. Visualisation Matplotlib : barres de probabilités par token.
  3. Comparaison DIxAI vs annotations humaines (si disponibles).

Lancement:
    python experiments/text/visualize_explanations.py
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas de display requis
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dixai.text import (
    TextDecisionInformationExplainer,
    BaselineType,
)
from dixai.text.data import load_sst2


# ---------------------------------------------------------------------------
# Visualisation HTML
# ---------------------------------------------------------------------------

def explanation_to_html(
    tokens,
    mask_probs,
    text,
    predicted_class,
    label_names=None,
    threshold=0.5,
    gold_mask=None,
):
    """
    Génère une visualisation HTML d'une explication DIxAI-Text.

    Args:
        tokens        : liste de strings (tokens BERT).
        mask_probs    : (T,) probabilités dans [0,1].
        text          : texte original.
        predicted_class: int, classe prédite.
        label_names   : dict {int: str} pour afficher le nom de la classe.
        threshold     : seuil de sélection.
        gold_mask     : (T,) masque gold humain (optionnel).

    Returns:
        html_str: string HTML.
    """
    if label_names is None:
        label_names = {}

    pred_name = label_names.get(predicted_class, str(predicted_class))

    def prob_to_color(p, is_gold=False):
        """Couleur de fond selon probabilité."""
        if is_gold:
            # Annotation humaine : bleu
            intensity = int(p * 200)
            return f"rgb(100, 100, {200 + intensity // 4})"
        else:
            # DIxAI : rouge si sélectionné, blanc sinon
            r = int(255)
            g = int(255 * (1 - p))
            b = int(255 * (1 - p))
            return f"rgb({r},{g},{b})"

    # CSS
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 2px;
            border-radius: 3px;
            font-size: 14px;
            border: 1px solid #ddd;
        }
        .selected { border: 2px solid #e74c3c; font-weight: bold; }
        .gold { border: 2px solid #3498db; }
        .both { border: 2px solid #8e44ad; font-weight: bold; }
        .legend { margin-top: 15px; font-size: 12px; }
        .legend span { padding: 3px 8px; border-radius: 3px; margin-right: 10px; }
        h3 { color: #2c3e50; }
        .meta { color: #7f8c8d; font-size: 12px; margin-bottom: 10px; }
    </style>
    """

    # En-tête
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">{css}</head>
<body>
<h3>DIxAI-Text — Explication</h3>
<div class="meta">
    <b>Texte :</b> {text}<br>
    <b>Classe prédite :</b> {pred_name}<br>
    <b>Threshold :</b> {threshold}
</div>
<div>
"""

    # Tokens colorés
    for t_idx, (tok, p) in enumerate(zip(tokens, mask_probs.tolist())):
        is_selected = p > threshold
        is_gold = (
            gold_mask is not None and
            t_idx < len(gold_mask) and
            gold_mask[t_idx].item() > 0.5
        )

        # Classe CSS
        if is_selected and is_gold:
            css_class = "token both"
            title = f"DIxAI={p:.3f} | Gold=1 (accord)"
        elif is_selected:
            css_class = "token selected"
            title = f"DIxAI={p:.3f} (sélectionné)"
        elif is_gold:
            css_class = "token gold"
            title = f"DIxAI={p:.3f} | Gold=1 (manqué)"
        else:
            css_class = "token"
            title = f"DIxAI={p:.3f}"

        bg_color = prob_to_color(p, is_gold=False)
        clean_tok = tok.replace("<", "&lt;").replace(">", "&gt;")

        html += (
            f'<span class="{css_class}" '
            f'style="background-color:{bg_color};" '
            f'title="{title}">'
            f'{clean_tok} <sub>{p:.2f}</sub>'
            f'</span>\n'
        )

    html += "</div>\n"

    # Légende
    html += """
<div class="legend">
    <b>Légende :</b>
    <span style="background:#ffaaaa; border:2px solid #e74c3c;">
        DIxAI sélectionné
    </span>
    <span style="background:#aaaaff; border:2px solid #3498db;">
        Annotation humaine
    </span>
    <span style="background:#ddaaff; border:2px solid #8e44ad;">
        Accord DIxAI + Humain
    </span>
</div>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Visualisation Matplotlib
# ---------------------------------------------------------------------------

def plot_token_probabilities(
    tokens,
    mask_probs,
    title="DIxAI-Text — Probabilités par token",
    threshold=0.5,
    gold_mask=None,
    save_path=None,
):
    """
    Barplot des probabilités par token.

    Args:
        tokens    : liste de strings.
        mask_probs: (T,) probabilités dans [0,1].
        title     : titre du graphique.
        threshold : seuil de sélection (ligne rouge).
        gold_mask : (T,) masque gold humain (optionnel).
        save_path : chemin de sauvegarde (None = affichage).
    """
    # Filtrer tokens spéciaux pour l'affichage
    display_tokens = []
    display_probs = []
    display_gold = []

    for t_idx, (tok, p) in enumerate(zip(tokens, mask_probs.tolist())):
        if tok in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]:
            continue
        display_tokens.append(tok.replace("##", ""))
        display_probs.append(p)
        if gold_mask is not None and t_idx < len(gold_mask):
            display_gold.append(gold_mask[t_idx].item())
        else:
            display_gold.append(0.0)

    T = len(display_tokens)
    if T == 0:
        return

    x = np.arange(T)
    fig, ax = plt.subplots(figsize=(max(8, T * 0.6), 4))

    # Couleurs des barres
    colors = []
    for p, g in zip(display_probs, display_gold):
        if p > threshold and g > 0.5:
            colors.append("#8e44ad")   # accord : violet
        elif p > threshold:
            colors.append("#e74c3c")   # DIxAI seulement : rouge
        elif g > 0.5:
            colors.append("#3498db")   # gold seulement : bleu
        else:
            colors.append("#bdc3c7")   # non sélectionné : gris

    bars = ax.bar(x, display_probs, color=colors, edgecolor="white", linewidth=0.5)

    # Ligne threshold
    ax.axhline(
        y=threshold, color="red", linestyle="--",
        linewidth=1.5, label=f"Threshold={threshold}"
    )

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Probabilité de sélection")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Légende
    patches = [
        mpatches.Patch(color="#e74c3c", label="DIxAI sélectionné"),
        mpatches.Patch(color="#bdc3c7", label="Non sélectionné"),
    ]
    if gold_mask is not None:
        patches += [
            mpatches.Patch(color="#3498db", label="Gold humain"),
            mpatches.Patch(color="#8e44ad", label="Accord DIxAI+Humain"),
        ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {save_path}")
    else:
        plt.savefig("explanation_plot.png", dpi=150, bbox_inches="tight")
        print("  Figure sauvegardée : explanation_plot.png")

    plt.close()


# ---------------------------------------------------------------------------
# Visualisation comparative : plusieurs méthodes
# ---------------------------------------------------------------------------

def plot_comparison(
    tokens,
    method_scores,
    title="Comparaison méthodes XAI",
    threshold=0.5,
    gold_mask=None,
    save_path=None,
):
    """
    Barplot comparatif de plusieurs méthodes XAI.

    Args:
        tokens       : liste de strings (tokens de contenu).
        method_scores: dict {method_name: (T,) scores normalisés [0,1]}.
        title        : titre.
        threshold    : seuil.
        gold_mask    : (T,) masque gold humain (optionnel).
        save_path    : chemin de sauvegarde.
    """
    methods = list(method_scores.keys())
    n_methods = len(methods)
    T = len(tokens)

    if T == 0 or n_methods == 0:
        return

    fig, axes = plt.subplots(
        n_methods, 1,
        figsize=(max(8, T * 0.5), 2.5 * n_methods),
        sharex=True,
    )
    if n_methods == 1:
        axes = [axes]

    method_colors = {
        "dixai": "#e74c3c",
        "integrated_grads": "#2ecc71",
        "attention": "#f39c12",
        "lime": "#9b59b6",
        "shap": "#1abc9c",
        "random": "#95a5a6",
    }

    x = np.arange(T)

    for ax, method in zip(axes, methods):
        scores = method_scores[method]
        # Normaliser dans [0,1] pour comparaison visuelle
        s_min, s_max = scores.min().item(), scores.max().item()
        if s_max > s_min:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = scores.clone()

        color = method_colors.get(method, "#3498db")
        ax.bar(x, scores_norm.tolist(), color=color, alpha=0.8, edgecolor="white")

        # Gold mask en overlay
        if gold_mask is not None:
            for t_idx, g in enumerate(gold_mask.tolist()):
                if g > 0.5 and t_idx < T:
                    ax.axvspan(
                        t_idx - 0.4, t_idx + 0.4,
                        alpha=0.2, color="#3498db",
                    )

        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel(method, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1.0])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    axes[0].set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure comparative sauvegardée : {save_path}")
    else:
        plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight")
        print("  Figure comparative sauvegardée : comparison_plot.png")

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    model_name = "textattack/bert-base-uncased-SST-2"

    print("Chargement modèle...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="eager",
        use_safetensors=True,
    ).to(device)
    model.eval()

    explainer = TextDecisionInformationExplainer(
        model, tokenizer,
        lambda_fidelity=10.0,
        lambda_contiguity=2.0,
        baseline_type=BaselineType.MASK_TOKEN,
        device="cpu",
    )

    # Dossier de sortie
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "visualizations"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Exemples SST-2
    examples = load_sst2(limit=4, split="validation", seed=42)
    label_names = {0: "NEGATIVE", 1: "POSITIVE"}

    print("\n" + "=" * 70)
    print("DIxAI-Text — Visualisation qualitative")
    print("=" * 70)

    all_html = []

    for i, ex in enumerate(examples):
        text = ex.text
        print(f"\n[{i+1}] {text[:60]}...")

        explanation = explainer.explain(
            text, steps=200, lr=0.2, verbose=False, seed=0
        )

        tokens = explanation.tokens
        mask_probs = explanation.mask_probs
        predicted_class = explanation.predicted_class

        print(f"  Classe prédite : {label_names.get(predicted_class, predicted_class)}")
        print(f"  Tokens sélectionnés : {explanation.top_tokens(threshold=0.5)}")

        # --- HTML ---
        html = explanation_to_html(
            tokens=tokens,
            mask_probs=mask_probs,
            text=text,
            predicted_class=predicted_class,
            label_names=label_names,
            threshold=0.5,
        )
        all_html.append(html)

        html_path = os.path.join(output_dir, f"explanation_{i+1}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  HTML sauvegardé : {html_path}")

        # --- Matplotlib barplot ---
        plot_token_probabilities(
            tokens=tokens,
            mask_probs=mask_probs,
            title=f'DIxAI-Text | "{text[:40]}..."',
            threshold=0.5,
            save_path=os.path.join(output_dir, f"barplot_{i+1}.png"),
        )

    # --- HTML global (toutes les explications) ---
    combined_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DIxAI-Text — Explications SST-2</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .explanation { border: 1px solid #ddd; padding: 15px;
                       margin-bottom: 20px; border-radius: 5px; }
        h1 { color: #2c3e50; }
    </style>
</head>
<body>
<h1>DIxAI-Text — Explications SST-2</h1>
"""
    for j, html in enumerate(all_html):
        # Extraire le body de chaque HTML
        body_start = html.find("<body>") + 6
        body_end = html.find("</body>")
        body_content = html[body_start:body_end]
        combined_html += f'<div class="explanation">{body_content}</div>\n'

    combined_html += "</body></html>"

    combined_path = os.path.join(output_dir, "all_explanations.html")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined_html)
    print(f"\nHTML global sauvegardé : {combined_path}")

    print("\n" + "=" * 70)
    print("Visualisations terminées.")
    print(f"Dossier de sortie : {output_dir}")


if __name__ == "__main__":
    main()
