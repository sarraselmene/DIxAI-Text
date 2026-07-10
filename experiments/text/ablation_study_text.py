"""
Ablation Study for DIxAI-Text
Teste la contribution de chaque composant spécifique au texte :

  1. Baseline B          : mask_token vs pad_token vs mean_embedding
  2. Pénalité de contiguïté : lambda_contiguity = 0.0 vs valeur nominale vs forte
  3. Annealing température  : avec vs sans annealing (température fixe)

Sauvegarde les résultats en CSV + génère un tableau LaTeX pour le rapport.

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

from dixai.text import TextDecisionInformationExplainer, BaselineType

torch.manual_seed(42)
np.random.seed(42)

MODEL_NAME = "textattack/bert-base-uncased-SST-2"

# Nombre de phrases à tirer du vrai split de validation SST-2 (glue/sst2).
N_SAMPLES = 60


def load_sst2_sentences(n_samples: int = N_SAMPLES, seed: int = 42):
    """
    Charge un échantillon aléatoire de phrases depuis le vrai split de
    validation SST-2 (glue/sst2 sur HuggingFace datasets), plutôt qu'une
    liste de phrases choisies à la main. Nécessaire pour des résultats
    statistiquement solides dans le rapport / une éventuelle soumission.

    Fallback : si `datasets` n'est pas installé ou le téléchargement échoue
    (pas de réseau), on retombe sur une petite liste de phrases manuelles.
    """
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
        print(f"Impossible de charger glue/sst2 ({e}). Fallback sur la liste manuelle.")
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


TEST_SENTENCES = load_sst2_sentences()


class AblationConfig:
    def __init__(self, name, baseline_type, lambda_contiguity, use_annealing,
                 lambda_fidelity=10.0, steps=200, lr=0.2, temperature=2.0 / 3.0):
        self.name = name
        self.baseline_type = baseline_type
        self.lambda_contiguity = lambda_contiguity
        self.use_annealing = use_annealing
        self.lambda_fidelity = lambda_fidelity
        self.steps = steps
        self.lr = lr
        self.temperature = temperature


# ============== Configurations d'ablation ==============
# Chaque config ne change qu'UNE variable par rapport à la config "Full" pour
# isoler proprement la contribution de chaque composant.
CONFIGS = [
    # --- Config de référence ---
    AblationConfig("Full DIxAI-Text (mask_token)", BaselineType.MASK_TOKEN,
                    lambda_contiguity=2.0, use_annealing=True),

    # --- Ablation 1 : choix de la baseline B ---
    AblationConfig("Baseline = pad_token", BaselineType.PAD_TOKEN,
                    lambda_contiguity=2.0, use_annealing=True),
    AblationConfig("Baseline = mean_embedding", BaselineType.MEAN_EMBEDDING,
                    lambda_contiguity=2.0, use_annealing=True),

    # --- Ablation 2 : pénalité de contiguïté ---
    AblationConfig("No Contiguity (lambda=0)", BaselineType.MASK_TOKEN,
                    lambda_contiguity=0.0, use_annealing=True),
    AblationConfig("Strong Contiguity (lambda=5)", BaselineType.MASK_TOKEN,
                    lambda_contiguity=5.0, use_annealing=True),

    # --- Ablation 3 : annealing de température ---
    AblationConfig("No Annealing (temp fixe)", BaselineType.MASK_TOKEN,
                    lambda_contiguity=2.0, use_annealing=False),
]


def run_single_explain(explainer, text, config):
    """Lance explain() ; si use_annealing=False, on désactive l'annealing du
    schedule interne pour isoler l'effet de la température fixe."""
    explanation = explainer.explain(
        text,
        steps=config.steps,
        lr=config.lr,
        temperature=config.temperature,
        anneal=config.use_annealing,
        verbose=False,
        seed=0,
    )
    return explanation


def run_ablation_study():
    print("=" * 70)
    print("DIxAI-Text Ablation Study")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"\nChargement du modèle {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, attn_implementation="eager", use_safetensors=True
    ).to(device)
    model.eval()

    results = []

    for config in CONFIGS:
        print(f"\n--- Running: {config.name} ---")

        fidelities, sparsities, contiguities = [], [], []
        per_sentence_records = []

        explainer = TextDecisionInformationExplainer(
            model,
            tokenizer,
            lambda_fidelity=config.lambda_fidelity,
            lambda_contiguity=config.lambda_contiguity,
            baseline_type=config.baseline_type,
            device="cpu",
        )

        for text in TEST_SENTENCES:
            explanation = run_single_explain(explainer, text, config)
            fidelities.append(explanation.fidelity_score)
            sparsities.append(explanation.sparsity_score)
            contiguities.append(explanation.contiguity_score)
            per_sentence_records.append({
                "text": text,
                "fidelity": explanation.fidelity_score,
                "sparsity": explanation.sparsity_score,
                "contiguity": explanation.contiguity_score,
            })

        avg_fid, std_fid = np.mean(fidelities), np.std(fidelities)
        avg_spar, std_spar = np.mean(sparsities), np.std(sparsities)
        avg_con, std_con = np.mean(contiguities), np.std(contiguities)

        # Identifie les 3 pires phrases (fidelity la plus mauvaise = KL la plus
        # haute) pour cette config, utile pour comprendre les gros écarts-types.
        worst = sorted(per_sentence_records, key=lambda r: r["fidelity"], reverse=True)[:3]
        print("  Pires phrases (fidelity la plus mauvaise) :")
        for w in worst:
            snippet = w["text"][:70] + ("..." if len(w["text"]) > 70 else "")
            print(f"    KL={w['fidelity']:.3f}  \"{snippet}\"")

        results.append({
            "Configuration": config.name,
            "Fidelity": f"{avg_fid:.3f} ± {std_fid:.3f}",
            "Sparsity": f"{avg_spar:.3f} ± {std_spar:.3f}",
            "Contiguity": f"{avg_con:.3f} ± {std_con:.3f}",
            "Fidelity_mean": avg_fid,
            "Sparsity_mean": avg_spar,
            "Contiguity_mean": avg_con,
        })

        # Sauvegarde le détail par phrase pour cette config (utile pour
        # investiguer les outliers plus tard sans tout relancer).
        detail_df = pd.DataFrame(per_sentence_records)
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        safe_name = config.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        detail_df.to_csv(os.path.join(results_dir, f"detail_{safe_name}.csv"), index=False)

        print(f"  Fidelity (KL, plus bas = mieux)  : {avg_fid:.3f} ± {std_fid:.3f}")
        print(f"  Sparsity (taux tokens gardés)     : {avg_spar:.3f} ± {std_spar:.3f}")
        print(f"  Contiguity (plus bas = spans compacts) : {avg_con:.3f} ± {std_con:.3f}")

    # Sauvegarde CSV
    df = pd.DataFrame(results)
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "ablation_study_text_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés : {output_path}")

    # Tableau LaTeX pour le rapport
    print("\n" + "=" * 70)
    print("LaTeX Table for Manuscript:")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Étude d'ablation : contribution des composants de DIxAI-Text}")
    print(r"\label{tab:ablation_text}")
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{Fidelity} & \textbf{Sparsity} & \textbf{Contiguity} \\ \midrule")
    for r in results:
        print(f"{r['Configuration']:32} & {r['Fidelity']:15} & {r['Sparsity']:15} & {r['Contiguity']:15} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return df


if __name__ == "__main__":
    run_ablation_study()