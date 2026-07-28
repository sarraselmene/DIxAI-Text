"""
Ablation Study for DIxAI-Text — avec métriques ERASER intégrées.

Teste :
  1. Baseline B          : mask_token vs pad_token vs mean_embedding
  2. Pénalité contiguité : lambda=0 vs nominale vs forte
  3. Annealing           : avec vs sans

Sauvegarde CSV + tableau LaTeX.

Lancement:
    python experiments/text/ablation_study_text.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dixai.text import (
    TextDecisionInformationExplainer,
    BaselineType,
    TextBaselineProvider,
)
from dixai.text.metrics_text import compute_eraser_scores

torch.manual_seed(42)
np.random.seed(42)

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
N_SAMPLES = 30  # réduit pour rapidité ; augmenter pour le paper final


def load_sst2_sentences(n_samples: int = N_SAMPLES, seed: int = 42):
    """Charge SST-2 validation depuis HuggingFace datasets (fallback manuel)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        rng = np.random.default_rng(seed)
        n = min(n_samples, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        sentences = [ds[int(i)]["sentence"].strip() for i in idx]
        print(f"Chargé {len(sentences)} phrases depuis glue/sst2 (validation).")
        return sentences
    except Exception as e:
        print(f"Fallback liste manuelle ({e}).")
        return [
            "This movie was not very good, honestly quite disappointing.",
            "An absolute masterpiece with brilliant performances.",
            "The plot was boring and the acting felt wooden.",
            "I really enjoyed every minute of this film.",
            "Waste of time, poorly written and badly directed.",
            "A charming, heartfelt story that stayed with me.",
            "Predictable and dull from start to finish.",
            "One of the best films I have seen this year.",
        ]


class AblationConfig:
    def __init__(
        self, name, baseline_type, lambda_contiguity,
        use_annealing, lambda_fidelity=10.0,
        steps=200, lr=0.2, temperature=2.0 / 3.0,
    ):
        self.name = name
        self.baseline_type = baseline_type
        self.lambda_contiguity = lambda_contiguity
        self.use_annealing = use_annealing
        self.lambda_fidelity = lambda_fidelity
        self.steps = steps
        self.lr = lr
        self.temperature = temperature


CONFIGS = [
    # Config de référence
    AblationConfig(
        "Full DIxAI-Text (mask_token)",
        BaselineType.MASK_TOKEN, lambda_contiguity=2.0, use_annealing=True,
    ),
    # Ablation 1 : baseline
    AblationConfig(
        "Baseline = pad_token",
        BaselineType.PAD_TOKEN, lambda_contiguity=2.0, use_annealing=True,
    ),
    AblationConfig(
        "Baseline = mean_embedding",
        BaselineType.MEAN_EMBEDDING, lambda_contiguity=2.0, use_annealing=True,
    ),
    # Ablation 2 : contiguité
    AblationConfig(
        "No Contiguity (lambda=0)",
        BaselineType.MASK_TOKEN, lambda_contiguity=0.0, use_annealing=True,
    ),
    AblationConfig(
        "Strong Contiguity (lambda=5)",
        BaselineType.MASK_TOKEN, lambda_contiguity=5.0, use_annealing=True,
    ),
    # Ablation 3 : annealing
    AblationConfig(
        "No Annealing (temp fixe)",
        BaselineType.MASK_TOKEN, lambda_contiguity=2.0, use_annealing=False,
    ),
]


def run_ablation_study():
    print("=" * 70)
    print("DIxAI-Text Ablation Study (avec métriques ERASER)")
    print("=" * 70)

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, attn_implementation="eager", use_safetensors=True
    ).to(device)
    model.eval()

    sentences = load_sst2_sentences()
    results = []

    for config in CONFIGS:
        print(f"\n--- Running: {config.name} ---")

        # Baseline provider pour ERASER (même type que l'explainer)
        baseline_provider = TextBaselineProvider(model, tokenizer, config.baseline_type)
        baseline_vec = baseline_provider.get_baseline_vector(device=device)

        explainer = TextDecisionInformationExplainer(
            model, tokenizer,
            lambda_fidelity=config.lambda_fidelity,
            lambda_contiguity=config.lambda_contiguity,
            baseline_type=config.baseline_type,
            device="cpu",
        )

        fidelities, sparsities, contiguities = [], [], []
        suff_list, comp_list = [], []
        per_sentence_records = []

        for text in sentences:
            explanation = explainer.explain(
                text,
                steps=config.steps,
                lr=config.lr,
                temperature=config.temperature,
                anneal=config.use_annealing,
                verbose=False,
                seed=0,
            )

            # Métriques intrinsèques
            fidelities.append(explanation.fidelity_score)
            sparsities.append(explanation.sparsity_score)
            contiguities.append(explanation.contiguity_score)

            # Métriques ERASER (threshold=0.5)
            eraser = compute_eraser_scores(
                model, tokenizer, text,
                mask_probs=explanation.mask_probs,
                baseline_vec=baseline_vec,
                device="cpu",
                strategy="threshold",
                threshold=0.5,
            )
            suff_list.append(eraser.sufficiency)
            comp_list.append(eraser.comprehensiveness)

            per_sentence_records.append({
                "text": text,
                "fidelity": explanation.fidelity_score,
                "sparsity": explanation.sparsity_score,
                "contiguity": explanation.contiguity_score,
                "sufficiency": eraser.sufficiency,
                "comprehensiveness": eraser.comprehensiveness,
                "n_tokens_kept": eraser.n_tokens_kept,
                "n_tokens_total": eraser.n_tokens_total,
            })

        avg_fid  = np.mean(fidelities);  std_fid  = np.std(fidelities)
        avg_spar = np.mean(sparsities);  std_spar = np.std(sparsities)
        avg_con  = np.mean(contiguities); std_con  = np.std(contiguities)
        avg_suff = np.mean(suff_list);   std_suff = np.std(suff_list)
        avg_comp = np.mean(comp_list);   std_comp = np.std(comp_list)

        print(f"  Fidelity (KL, bas=mieux)          : {avg_fid:.3f} ± {std_fid:.3f}")
        print(f"  Sparsity (soft, tokens gardés)    : {avg_spar:.3f} ± {std_spar:.3f}")
        print(f"  Contiguity (TV_seq, bas=mieux)    : {avg_con:.3f} ± {std_con:.3f}")
        print(f"  Sufficiency ERASER (bas=mieux)    : {avg_suff:.3f} ± {std_suff:.3f}")
        print(f"  Comprehensiveness ERASER (haut=mieux): {avg_comp:.3f} ± {std_comp:.3f}")

        results.append({
            "Configuration": config.name,
            "Fidelity":          f"{avg_fid:.3f} ± {std_fid:.3f}",
            "Sparsity":          f"{avg_spar:.3f} ± {std_spar:.3f}",
            "Contiguity":        f"{avg_con:.3f} ± {std_con:.3f}",
            "Sufficiency":       f"{avg_suff:.3f} ± {std_suff:.3f}",
            "Comprehensiveness": f"{avg_comp:.3f} ± {std_comp:.3f}",
            # valeurs numériques pour tri/analyse
            "Fidelity_mean":          avg_fid,
            "Sparsity_mean":          avg_spar,
            "Contiguity_mean":        avg_con,
            "Sufficiency_mean":       avg_suff,
            "Comprehensiveness_mean": avg_comp,
        })

        # Sauvegarde détail par phrase
        results_dir = os.path.join(
            os.path.dirname(__file__), "..", "results"
        )
        os.makedirs(results_dir, exist_ok=True)
        safe_name = (
            config.name.lower()
            .replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        )
        pd.DataFrame(per_sentence_records).to_csv(
            os.path.join(results_dir, f"detail_{safe_name}.csv"), index=False
        )

    # Sauvegarde CSV global
    df = pd.DataFrame(results)
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "ablation_study_text_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés : {output_path}")

    # Tableau LaTeX
    print("\n" + "=" * 70)
    print("LaTeX Table:")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study DIxAI-Text sur SST-2 (N=" + str(len(sentences)) + r").}")
    print(r"\label{tab:ablation_text}")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(
        r"\textbf{Config} & \textbf{Fidelity}$\downarrow$ & "
        r"\textbf{Sparsity} & \textbf{Contiguity}$\downarrow$ & "
        r"\textbf{Suff}$\downarrow$ & \textbf{Comp}$\uparrow$ \\ \midrule"
    )
    for r in results:
        print(
            f"{r['Configuration']} & {r['Fidelity']} & {r['Sparsity']} & "
            f"{r['Contiguity']} & {r['Sufficiency']} & {r['Comprehensiveness']} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return df


if __name__ == "__main__":
    run_ablation_study()
