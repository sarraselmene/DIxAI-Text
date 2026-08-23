"""
pub_sst2.py — VERSION CORRIGÉE (protocole final + timing)
- Comparaison: N=50, steps=200, 3 seeds (aligné sur l'ablation)
- Préserve l'ablation existante dans le JSON (ne l'écrase plus)
- Ajoute le chronométrage (temps d'explication + inférence pure)
"""

import sys,os,json,time,hashlib
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","..","src"))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"]="1"

import torch,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoTokenizer,AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer,BaselineType,TextBaselineProvider
from dixai.text.metrics_text import compute_eraser_scores
from dixai.text.data import load_sst2

device="cpu"

# --- Ablation (déjà faite, on ne la relance pas) ---
N=50;STEPS=200;SEEDS=[0,1,2]

# --- Comparaison : PROTOCOLE FINAL, actuellement en MODE TEST ---
# >>> TEST DE TIMING (étape actuelle) <
#N_COMP=3;STEPS_COMP=200;SEEDS_COMP=[0]
#LIME_SAMPLES=500;SHAP_EVALS=200
# >>> RUN FINAL : remplacer la ligne ci-dessus par : <
N_COMP=50;STEPS_COMP=200;SEEDS_COMP=[0,1,2]
LIME_SAMPLES=500;SHAP_EVALS=100   #(ajuster selon résultat du test)

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results")
os.makedirs(RESULTS_DIR,exist_ok=True)
RESULTS_FILE=os.path.join(RESULTS_DIR,"pub_sst2.json")

print(f"=== PUB SST-2 | Comparaison: N={N_COMP}/steps={STEPS_COMP}/seeds={SEEDS_COMP} ===");sys.stdout.flush()

# ============================================================
# Charger modèle
# ============================================================
print("Chargement modèle...");sys.stdout.flush()
tok=AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
mdl=AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2",attn_implementation="eager",use_safetensors=True).to(device)
mdl.eval();emb=mdl.get_input_embeddings()

def get_cm(t,text,d):
    e=t(text,return_tensors="pt",truncation=True);ids=e["input_ids"].to(d);am=e["attention_mask"].to(d)
    sp=t.get_special_tokens_mask(ids[0].tolist(),already_has_special_tokens=True)
    stm=torch.tensor(sp,device=d,dtype=am.dtype).unsqueeze(0);cm=(am*(1-stm)).float().squeeze(0)
    return ids,am,cm

def topk(s,cm,k):
    s=s.clone().float();s[cm==0]=-1e9;idx=torch.argsort(s,descending=True)[:k]
    m=torch.zeros_like(cm,dtype=torch.float32);m[idx]=1.0;return m

def safe_mean(values, default=0.0):
    if not values: return default
    arr=np.asarray(values,dtype=float)
    if arr.size == 0 or np.isnan(arr).all(): return default
    return float(np.mean(arr))

def safe_std(values, default=0.0):
    if not values: return default
    arr=np.asarray(values,dtype=float)
    if arr.size == 0 or np.isnan(arr).all(): return default
    return float(np.std(arr))

# ============================================================
# FIX #1 : préserver l'ablation existante au lieu de l'écraser
# ============================================================
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE,"r") as f:
        saved=json.load(f)
    all_abl=saved.get("ablation",[])
    print(f"[INFO] Ablation existante préservée : {len(all_abl)} entrées");sys.stdout.flush()
else:
    all_abl=[]
    print("[INFO] Aucun résultat d'ablation existant trouvé (all_abl=[])");sys.stdout.flush()

# La comparaison, elle, est toujours relancée à neuf
comp_res={}

# ============================================================
# TIMING : inférence pure du classifieur (une passe forward)
# ============================================================
_probe_text="This is a sample sentence used to measure pure inference time."
_e=tok(_probe_text,return_tensors="pt")
with torch.no_grad():
    for _ in range(3):  # warmup
        mdl(**_e)
    t0=time.perf_counter()
    for _ in range(20):
        mdl(**_e)
    t_infer=(time.perf_counter()-t0)/20
print(f"[TIMING] Inférence pure (1 forward pass) ≈ {t_infer*1000:.2f} ms");sys.stdout.flush()

# ============================================================
# PARTIE 2 : COMPARISON (6 méthodes × N_COMP × SEEDS_COMP)
# ============================================================
methods=["dixai","random","integrated_grads","attention","lime","shap"]
comp_res={m:{"s":[],"c":[]} for m in methods}
explain_times=[]  # temps de génération d'une explication DIxAI (steps d'optimisation)

print("\n=== COMPARISON ===");sys.stdout.flush()

bp=TextBaselineProvider(mdl,tok,"mean_embedding");bv=bp.get_baseline_vector(device=device)

for seed in SEEDS_COMP:
    torch.manual_seed(seed);np.random.seed(seed)
    exs=load_sst2(limit=N_COMP,split="validation",seed=seed)
    ex_dixai=TextDecisionInformationExplainer(mdl,tok,lambda_fidelity=30.0,lambda_contiguity=2.0,baseline_type="mean_embedding",device=device)

    for i,e in enumerate(exs):
        if i%5==0:print(f"  seed={seed} [{i+1}/{N_COMP}]");sys.stdout.flush()

        t0=time.perf_counter()
        ep=ex_dixai.explain(e.text,steps=STEPS_COMP,lr=0.1,init_logits=-1.0,seed=seed,anneal=True)
        t_explain=time.perf_counter()-t0
        explain_times.append(t_explain)

        tgt=ep.predicted_class;ids,am,cm=get_cm(tok,e.text,device);T=ids.shape[1]
        mk=max(1,int(((ep.mask_probs>0.5).float()*cm).sum().item()))
        er=compute_eraser_scores(mdl,tok,e.text,ep.mask_probs,bv,device=device,strategy="threshold",threshold=0.5)
        comp_res["dixai"]["s"].append(float(er.sufficiency));comp_res["dixai"]["c"].append(float(er.comprehensiveness))

        rm=topk(torch.rand(T),cm,mk)
        er=compute_eraser_scores(mdl,tok,e.text,rm,bv,device=device,strategy="threshold",threshold=0.5)
        comp_res["random"]["s"].append(float(er.sufficiency));comp_res["random"]["c"].append(float(er.comprehensiveness))

        try:
            embeds=emb(ids);bl=torch.zeros_like(embeds);tg=torch.zeros_like(embeds)
            for alpha in torch.linspace(0,1,32):
                x=(bl+alpha*(embeds-bl)).clone().detach().requires_grad_(True)
                out=mdl(inputs_embeds=x,attention_mask=am);sc=out.logits[0,tgt]
                tg+=torch.autograd.grad(sc,x)[0].detach()
            ig=(embeds.detach()-bl)*tg/32;igs=ig.abs().sum(-1).squeeze(0)
            im=topk(igs,cm,mk)
            er=compute_eraser_scores(mdl,tok,e.text,im,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["integrated_grads"]["s"].append(float(er.sufficiency));comp_res["integrated_grads"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] IG failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

        try:
            with torch.no_grad():out=mdl(input_ids=ids,attention_mask=am,output_attentions=True)
            atts=getattr(out,"attentions",None)
            if atts:
                cls_row=atts[-1][0].mean(0)[0].detach();am2=topk(cls_row,cm,mk)
                er=compute_eraser_scores(mdl,tok,e.text,am2,bv,device=device,strategy="threshold",threshold=0.5)
                comp_res["attention"]["s"].append(float(er.sufficiency));comp_res["attention"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] Attention failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

        try:
            from lime.lime_text import LimeTextExplainer
            def pf(texts):
                enc=tok(texts,return_tensors="pt",truncation=True,padding=True)
                enc={k:v.to(device) for k,v in enc.items()}
                with torch.no_grad():out=mdl(**enc);probs=torch.softmax(out.logits,dim=-1)
                return probs.cpu().numpy()
            le=LimeTextExplainer(class_names=["0","1"]).explain_instance(e.text,pf,num_features=50,num_samples=LIME_SAMPLES,labels=[tgt])
            ws=dict(le.as_list(label=tgt))
            ls=torch.zeros(T);tokens=tok.convert_ids_to_tokens(ids[0].tolist())
            for ti,ts in enumerate(tokens):
                cl=ts.replace("##","",).lower().strip()
                for w,sc in ws.items():
                    if cl and (cl in w.lower() or w.lower() in cl):ls[ti]=max(ls[ti].item(),abs(sc))
            lm=topk(ls,cm,mk)
            er=compute_eraser_scores(mdl,tok,e.text,lm,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["lime"]["s"].append(float(er.sufficiency));comp_res["lime"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] LIME failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

        try:
            import shap
            def pf_shap(texts):
                if isinstance(texts,np.ndarray):texts=texts.tolist()
                enc=tok(texts,return_tensors="pt",truncation=True,padding=True)
                enc={k:v.to(device) for k,v in enc.items()}
                with torch.no_grad():out=mdl(**enc);probs=torch.softmax(out.logits,dim=-1)
                return probs.cpu().numpy()
            masker=shap.maskers.Text(tok)
            shap_ex=shap.Explainer(pf_shap,masker,output_names=["0","1"])
            sv=shap_ex([e.text],max_evals=SHAP_EVALS,batch_size=8)
            vals=np.abs(sv.values[0,:,tgt])
            enc2=tok(e.text,return_tensors="pt",truncation=True);T2=enc2["input_ids"].shape[1]
            ss_scores=torch.zeros(T2)
            if len(vals)>=T2:ss_scores=torch.tensor(vals[:T2],dtype=torch.float32)
            else:ss_scores[:len(vals)]=torch.tensor(vals,dtype=torch.float32)
            sm=topk(ss_scores,cm,mk)
            er=compute_eraser_scores(mdl,tok,e.text,sm,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["shap"]["s"].append(float(er.sufficiency));comp_res["shap"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] SHAP failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

    for m in methods:
        print(f"  [{m}] values={len(comp_res[m]['c'])} suff={len(comp_res[m]['s'])}");sys.stdout.flush()

timing_stats={
    "explain_mean_s":float(np.mean(explain_times)) if explain_times else None,
    "explain_std_s":float(np.std(explain_times)) if explain_times else None,
    "pure_inference_ms":t_infer*1000,
    "n_timed":len(explain_times),
}
print(f"\n[TIMING] Génération d'explication : {timing_stats['explain_mean_s']:.3f}s ± {timing_stats['explain_std_s']:.3f}s (n={timing_stats['n_timed']})");sys.stdout.flush()

with open(RESULTS_FILE,"w") as f:
    json.dump({"ablation":all_abl,"comparison":comp_res,"timing":timing_stats},f,indent=2)
print(f"\n  Comparison sauvegardée dans {RESULTS_FILE}");sys.stdout.flush()

# ============================================================
# PARTIE 5 : TABLE LATEX COMPARAISON
# ============================================================
print("\n=== TABLE LATEX COMPARAISON ===");sys.stdout.flush()
method_labels=["Random","Attention","LIME","SHAP","IG","DIxAI-Text"]
method_keys=["random","attention","lime","shap","integrated_grads","dixai"]

with open(os.path.join(RESULTS_DIR,"table_comparison.tex"),"w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{ERASER faithfulness comparison on SST-2 (N=%d, %d seeds).}\n" % (N_COMP,len(SEEDS_COMP)))
    f.write("\\begin{tabular}{lcc}\n\\toprule\n")
    f.write("Method & Sufficiency$\\downarrow$ & Comprehensiveness$\\uparrow$ \\\\\n\\midrule\n")
    for m,name in zip(method_keys,method_labels):
        s=comp_res.get(m,{}).get("s",[])
        c=comp_res.get(m,{}).get("c",[])
        sm=safe_mean(s, default=float("nan")) if s else float("nan")
        cm_=safe_mean(c, default=float("nan")) if c else float("nan")
        f.write(f"{name} & {sm:.4f} & {cm_:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\label{tab:comparison}\n\\end{table}\n")
print(f"  Table LaTeX sauvée : {RESULTS_DIR}/table_comparison.tex");sys.stdout.flush()

print("\n=== TERMINÉ ===");sys.stdout.flush()
