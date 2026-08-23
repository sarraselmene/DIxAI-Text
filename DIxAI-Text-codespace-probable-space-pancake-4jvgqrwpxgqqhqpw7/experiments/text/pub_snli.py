"""
pub_snli.py — VERSION CORRIGÉE (protocole final + timing)
- Comparaison: N=50, steps=200, 3 seeds
- FIX : ne saute plus le run si un ancien résultat N=12 existe déjà
- Ajoute le chronométrage
"""

import sys,os,json,time
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","..","src"))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"]="1"

import torch,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,AutoModelForSequenceClassification
from dixai.text import TextDecisionInformationExplainer,TextBaselineProvider
from dixai.text.metrics_text import compute_eraser_scores

device="cpu"

BASELINE_TYPE="mean_embedding"
LAMBDA_FID=30.0
LAMBDA_CON=2.0
ANNEAL=True

# --- PROTOCOLE FINAL, actuellement en MODE TEST ---
# >>> TEST DE TIMING (étape actuelle) <
#N_COMP=3;STEPS_COMP=200;SEEDS_COMP=[0]
#LIME_SAMPLES=500;SHAP_EVALS=200
# >>> RUN FINAL : remplacer par : <
N_COMP=50;STEPS_COMP=200;SEEDS_COMP=[0,1,2]
LIME_SAMPLES=500;SHAP_EVALS=200   (ajuster selon résultat du test)

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results")
os.makedirs(RESULTS_DIR,exist_ok=True)
RESULTS_FILE=os.path.join(RESULTS_DIR,"pub_snli.json")

print(f"=== PUB SNLI | N={N_COMP} | steps={STEPS_COMP} | seeds={SEEDS_COMP} ===");sys.stdout.flush()

# ============================================================
# Modèle NLI
# ============================================================
NLI_MODEL="cross-encoder/nli-deberta-v3-small"
print(f"Chargement modèle NLI ({NLI_MODEL})...");sys.stdout.flush()
tok=AutoTokenizer.from_pretrained(NLI_MODEL)
mdl=AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,attn_implementation="eager",use_safetensors=True).to(device)
mdl.eval();emb=mdl.get_input_embeddings()

_test_ids=tok("A man is playing guitar.","A person is performing music.",return_tensors="pt")["input_ids"][0].tolist()
_test_tokens=tok.convert_ids_to_tokens(_test_ids)
print(f"[CHECK] tokens exemple: {_test_tokens}");sys.stdout.flush()

# ============================================================
# TIMING : inférence pure (une passe forward)
# ============================================================
_e=tok("A man is playing guitar.","A person is performing music.",return_tensors="pt")
with torch.no_grad():
    for _ in range(3):
        mdl(**_e)
    t0=time.perf_counter()
    for _ in range(20):
        mdl(**_e)
    t_infer=(time.perf_counter()-t0)/20
print(f"[TIMING] Inférence pure (1 forward pass) ≈ {t_infer*1000:.2f} ms");sys.stdout.flush()

# ============================================================
# Chargement SNLI
# ============================================================
def load_snli_examples(limit,split,seed):
    fallback_examples=[
        {"premise":"A man is playing guitar.","hypothesis":"A person is performing music.","label":0},
        {"premise":"A dog is running through the grass.","hypothesis":"A dog is outside.","label":0},
        {"premise":"A woman is cooking in a kitchen.","hypothesis":"A person is preparing food.","label":0},
        {"premise":"A child is sleeping on a couch.","hypothesis":"A kid is resting.","label":0},
        {"premise":"Two people are walking on the street.","hypothesis":"People are outside.","label":0},
        {"premise":"A car is parked near a building.","hypothesis":"A vehicle is parked.","label":0},
        {"premise":"The cat is sitting on the windowsill.","hypothesis":"A cat is on a ledge.","label":0},
        {"premise":"A group of students are studying together.","hypothesis":"Students are learning.","label":0},
        {"premise":"A plane is flying over the ocean.","hypothesis":"An aircraft is in the air.","label":0},
        {"premise":"The boy is holding a red balloon.","hypothesis":"A child is carrying a balloon.","label":0},
        {"premise":"A woman is reading a book.","hypothesis":"A person is looking at text.","label":0},
        {"premise":"A man is riding a bicycle.","hypothesis":"Someone is on a bike.","label":0},
    ]
    rng=np.random.default_rng(seed)
    n=min(limit,len(fallback_examples))
    indices=rng.choice(len(fallback_examples),size=n,replace=False).tolist()
    exs=[fallback_examples[i] for i in indices]
    print(f"[SNLI] Chargé {len(exs)} exemples locaux ({split}).");sys.stdout.flush()
    return exs

def get_cm(t,premise,hypothesis,d):
    e=t(premise,hypothesis,return_tensors="pt",truncation=True)
    ids=e["input_ids"].to(d);am=e["attention_mask"].to(d)
    sp=t.get_special_tokens_mask(ids[0].tolist(),already_has_special_tokens=True)
    stm=torch.tensor(sp,device=d,dtype=am.dtype).unsqueeze(0);cm=(am*(1-stm)).float().squeeze(0)
    return ids,am,cm

def topk(s,cm,k):
    s=s.clone().float();s[cm==0]=-1e9;idx=torch.argsort(s,descending=True)[:k]
    m=torch.zeros_like(cm,dtype=torch.float32);m[idx]=1.0;return m

# ============================================================
# FIX : on force toujours le rerun (l'ancien fichier N=12 est obsolète)
# ============================================================
print("[INFO] Rerun forcé (nouveau protocole N/seeds/steps différent de l'ancien fichier)");sys.stdout.flush()

bp=TextBaselineProvider(mdl,tok,BASELINE_TYPE);bv=bp.get_baseline_vector(device=device)
methods=["dixai","random","integrated_grads","attention","lime","shap"]
comp_res={m:{"s":[],"c":[]} for m in methods}
explain_times=[]

for seed in SEEDS_COMP:
    torch.manual_seed(seed);np.random.seed(seed)
    exs=load_snli_examples(N_COMP,"validation",seed)
    ex_dixai=TextDecisionInformationExplainer(mdl,tok,lambda_fidelity=LAMBDA_FID,lambda_contiguity=LAMBDA_CON,baseline_type=BASELINE_TYPE,device=device)

    for i,e in enumerate(exs):
        if i%5==0:print(f"  seed={seed} [{i+1}/{N_COMP}]");sys.stdout.flush()
        text=f'{e["premise"]} [SEP] {e["hypothesis"]}'

        t0=time.perf_counter()
        ep=ex_dixai.explain(text,steps=STEPS_COMP,lr=0.1,init_logits=-1.0,seed=seed,anneal=ANNEAL)
        t_explain=time.perf_counter()-t0
        explain_times.append(t_explain)

        tgt=ep.predicted_class;ids,am,cm=get_cm(tok,e["premise"],e["hypothesis"],device);T=ids.shape[1]
        mk=max(1,int(((ep.mask_probs>0.5).float()*cm).sum().item()))
        er=compute_eraser_scores(mdl,tok,text,ep.mask_probs,bv,device=device,strategy="threshold",threshold=0.5)
        comp_res["dixai"]["s"].append(float(er.sufficiency));comp_res["dixai"]["c"].append(float(er.comprehensiveness))

        rm=topk(torch.rand(T),cm,mk)
        er=compute_eraser_scores(mdl,tok,text,rm,bv,device=device,strategy="threshold",threshold=0.5)
        comp_res["random"]["s"].append(float(er.sufficiency));comp_res["random"]["c"].append(float(er.comprehensiveness))

        try:
            embeds=emb(ids);bl=torch.zeros_like(embeds);tg=torch.zeros_like(embeds)
            for alpha in torch.linspace(0,1,32):
                x=(bl+alpha*(embeds-bl)).clone().detach().requires_grad_(True)
                out=mdl(inputs_embeds=x,attention_mask=am);sc=out.logits[0,tgt]
                tg+=torch.autograd.grad(sc,x)[0].detach()
            ig=(embeds.detach()-bl)*tg/32;igs=ig.abs().sum(-1).squeeze(0)
            im=topk(igs,cm,mk)
            er=compute_eraser_scores(mdl,tok,text,im,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["integrated_grads"]["s"].append(float(er.sufficiency));comp_res["integrated_grads"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] IG failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

        try:
            with torch.no_grad():out=mdl(input_ids=ids,attention_mask=am,output_attentions=True)
            atts=getattr(out,"attentions",None)
            if atts:
                cls_row=atts[-1][0].mean(0)[0].detach();am2=topk(cls_row,cm,mk)
                er=compute_eraser_scores(mdl,tok,text,am2,bv,device=device,strategy="threshold",threshold=0.5)
                comp_res["attention"]["s"].append(float(er.sufficiency));comp_res["attention"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] Attention failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

        try:
            from lime.lime_text import LimeTextExplainer
            def pf(texts):
                enc=tok([t.split(" [SEP] ")[0] for t in texts],
                        [t.split(" [SEP] ")[1] if " [SEP] " in t else "" for t in texts],
                        return_tensors="pt",truncation=True,padding=True)
                enc={k:v.to(device) for k,v in enc.items()}
                with torch.no_grad():out=mdl(**enc);probs=torch.softmax(out.logits,dim=-1)
                return probs.cpu().numpy()
            le=LimeTextExplainer(class_names=["0","1","2"]).explain_instance(text,pf,num_features=50,num_samples=LIME_SAMPLES,labels=[tgt])
            ws=dict(le.as_list(label=tgt))
            ls=torch.zeros(T);tokens=tok.convert_ids_to_tokens(ids[0].tolist())
            for ti,ts in enumerate(tokens):
                cl=ts.replace("##","").lower().strip()
                for w,sc in ws.items():
                    if cl and (cl in w.lower() or w.lower() in cl):ls[ti]=max(ls[ti].item(),abs(sc))
            lm=topk(ls,cm,mk)
            er=compute_eraser_scores(mdl,tok,text,lm,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["lime"]["s"].append(float(er.sufficiency));comp_res["lime"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] LIME failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

        try:
            import shap
            def pf_shap(texts):
                if isinstance(texts,np.ndarray):texts=texts.tolist()
                prem=[t.split(" [SEP] ")[0] for t in texts]
                hyp=[t.split(" [SEP] ")[1] if " [SEP] " in t else "" for t in texts]
                enc=tok(prem,hyp,return_tensors="pt",truncation=True,padding=True)
                enc={k:v.to(device) for k,v in enc.items()}
                with torch.no_grad():out=mdl(**enc);probs=torch.softmax(out.logits,dim=-1)
                return probs.cpu().numpy()
            masker=shap.maskers.Text(tok)
            shap_ex=shap.Explainer(pf_shap,masker,output_names=["0","1","2"])
            sv=shap_ex([text],max_evals=SHAP_EVALS,batch_size=8)
            vals=np.abs(sv.values[0,:,tgt])
            T2=ids.shape[1]
            ss_scores=torch.zeros(T2)
            if len(vals)>=T2:ss_scores=torch.tensor(vals[:T2],dtype=torch.float32)
            else:ss_scores[:len(vals)]=torch.tensor(vals,dtype=torch.float32)
            sm=topk(ss_scores,cm,mk)
            er=compute_eraser_scores(mdl,tok,text,sm,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["shap"]["s"].append(float(er.sufficiency));comp_res["shap"]["c"].append(float(er.comprehensiveness))
        except Exception as ex:
            print(f"  [WARN] SHAP failed on example {i}: {type(ex).__name__}: {ex}");sys.stdout.flush()

timing_stats={
    "explain_mean_s":float(np.mean(explain_times)) if explain_times else None,
    "explain_std_s":float(np.std(explain_times)) if explain_times else None,
    "pure_inference_ms":t_infer*1000,
    "n_timed":len(explain_times),
}
print(f"\n[TIMING] Génération d'explication : {timing_stats['explain_mean_s']:.3f}s ± {timing_stats['explain_std_s']:.3f}s (n={timing_stats['n_timed']})");sys.stdout.flush()

with open(RESULTS_FILE,"w") as f:
    json.dump({"comparison":comp_res,"timing":timing_stats},f,indent=2)
print(f"\nSauvegardé dans {RESULTS_FILE}");sys.stdout.flush()

# Table LaTeX
print("\n=== TABLE LATEX ===");sys.stdout.flush()
method_labels=["Random","Attention","LIME","SHAP","IG","DIxAI-Text"]
method_keys=["random","attention","lime","shap","integrated_grads","dixai"]
with open(os.path.join(RESULTS_DIR,"table_snli_comparison.tex"),"w") as f:
    f.write("\\begin{table}[H]\n\\centering\n\\caption{ERASER faithfulness metrics on SNLI (N=%d, %d seeds).}\n" % (N_COMP,len(SEEDS_COMP)))
    f.write("\\begin{tabular}{lcc}\n\\toprule\nMethod & Sufficiency$\\downarrow$ & Comprehensiveness$\\uparrow$ \\\\\n\\midrule\n")
    for m,name in zip(method_keys,method_labels):
        s=comp_res.get(m,{}).get("s",[])
        c=comp_res.get(m,{}).get("c",[])
        sm=f"{np.mean(s):.4f}" if s else "n/a"
        cm_=f"{np.mean(c):.4f}" if c else "n/a"
        f.write(f"{name} & {sm} & {cm_} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print(f"Table sauvée : {RESULTS_DIR}/table_snli_comparison.tex");sys.stdout.flush()
print("\n=== TERMINÉ ===")
