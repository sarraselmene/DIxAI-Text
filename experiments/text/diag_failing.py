import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer, BaselineType

device = torch.device("cpu")
MODEL_NAME = "textattack/bert-base-uncased-SST-2"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, attn_implementation="eager", use_safetensors=True,
).to(device)
model.eval()

sentences = [
    "brilliant performances and a gripping story.",
    "I was not impressed at all.",
]

explainer = TextDecisionInformationExplainer(
    model, tok, lambda_fidelity=2.0, lambda_contiguity=2.0,
    baseline_type=BaselineType.MASK_TOKEN, device="cpu",
)

for s in sentences:
    enc = tok(s, return_tensors="pt").to(device)
    with torch.no_grad():
        y_orig = model(**enc).logits
        probs_orig = F.softmax(y_orig, dim=-1)

        # baseline complète : tous les tokens remplacés par [MASK]
        mask_id = tok.mask_token_id
        full_mask_ids = enc["input_ids"].clone()
        full_mask_ids[:, 1:-1] = mask_id  # garde CLS/SEP, masque le reste
        y_null = model(input_ids=full_mask_ids, attention_mask=enc["attention_mask"]).logits
        probs_null = F.softmax(y_null, dim=-1)

    kl = F.kl_div(F.log_softmax(y_null, dim=-1), F.log_softmax(y_orig, dim=-1),
                   reduction="batchmean", log_target=True).item()

    print(f"\n>>> {s}")
    print("y_orig probs:", probs_orig.tolist())
    print("y_null probs (tout masqué):", probs_null.tolist())
    print(f"KL(y_orig, y_null) = {kl:.4f}")
