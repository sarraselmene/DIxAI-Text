"""
pub_sst2.py — VERSION COMPLÈTE
- N=50, steps=200, 3 seeds
- Sauvegarde résultats dans results/pub_sst2.json + CSV + figures PNG
- Si résultats existent déjà, les charge sans relancer
- Génère les 2 tables LaTeX + figures
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
N=50;STEPS=200;SEEDS=[0,1,2]

# Dossier résultats
RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results")
os.makedirs(RESULTS_DIR,exist_ok=True)
RESULTS_FILE=os.path.join(RESULTS_DIR,"pub_sst2.json")

print(f"=== PUB SST-2 | N={N} | steps={STEPS} | seeds={SEEDS} ===");sys.stdout.flush()

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

# ============================================================
# Vérifier si résultats existent déjà
# ============================================================
if os.path.exists(RESULTS_FILE):
    print(f"\nRésultats existants trouvés dans {RESULTS_FILE}")
    print("Chargement sans relancer les expériences...");sys.stdout.flush()
    with open(RESULTS_FILE,"r") as f:
        saved=json.load(f)
    all_abl=saved.get("ablation",[])
    comp_res=saved.get("comparison",{})
    # Convertir listes
    for m in comp_res:
        if isinstance(comp_res[m].get("s"),list):
            comp_res[m]["s"]=[float(x) for x in comp_res[m]["s"]]
            comp_res[m]["c"]=[float(x) for x in comp_res[m]["c"]]
else:
    all_abl=[]
    comp_res={}

# ============================================================
# PARTIE 1 : ABLATION (6 configs × N × 3 seeds)
# ============================================================
if not all_abl:
    print("\n=== ABLATION ===");sys.stdout.flush()

    abl_cfgs=[
        #("Full DIxAI-Text (mean_emb)","mean_embedding",30.0,2.0,True),
        #("Baseline=mask_token","mask_token",30.0,2.0,True),
        ("Baseline=pad_token","pad_token",30.0,2.0,True),
        ("No Contiguity","mean_embedding",30.0,0.0,True),
        ("Strong Contiguity","mean_embedding",30.0,5.0,True),
        ("No Annealing","mean_embedding",30.0,2.0,False),
    ]

    for cfg_name,bl,lf,lc,an in abl_cfgs:
        print(f"\n  {cfg_name}");sys.stdout.flush()
        t0=time.time()
        for seed in SEEDS:
            torch.manual_seed(seed);np.random.seed(seed)
            exs=load_sst2(limit=N,split="validation",seed=seed)
            ex=TextDecisionInformationExplainer(mdl,tok,lambda_fidelity=lf,lambda_contiguity=lc,baseline_type=bl,device=device)
            bp=TextBaselineProvider(mdl,tok,bl);bv=bp.get_baseline_vector(device=device)
            fs,ss,cs,su,co=[],[],[],[],[]
            for e in exs:
                ep=ex.explain(e.text,steps=STEPS,lr=0.1,init_logits=-1.0,seed=seed)
                er=compute_eraser_scores(mdl,tok,e.text,ep.mask_probs,bv,device=device,strategy="threshold",threshold=0.5)
                fs.append(ep.fidelity_score);ss.append(ep.sparsity_score);cs.append(ep.contiguity_score)
                su.append(er.sufficiency);co.append(er.comprehensiveness)
            all_abl.append({"config":cfg_name,"seed":seed,"n":len(fs),
                "fid_m":float(np.mean(fs)),"fid_s":float(np.std(fs)),
                "spar_m":float(np.mean(ss)),"spar_s":float(np.std(ss)),
                "cont_m":float(np.mean(cs)),"cont_s":float(np.std(cs)),
                "suff_m":float(np.mean(su)),"suff_s":float(np.std(su)),
                "comp_m":float(np.mean(co)),"comp_s":float(np.std(co))})
        dt=time.time()-t0
        rows=[r for r in all_abl if r["config"]==cfg_name]
        print(f"    {dt:.0f}s | Fid={np.mean([r['fid_m'] for r in rows]):.4f} | Suff={np.mean([r['suff_m'] for r in rows]):.4f} | Comp={np.mean([r['comp_m'] for r in rows]):.4f}");sys.stdout.flush()

    # Sauvegarder après ablation
    with open(RESULTS_FILE,"w") as f:
        json.dump({"ablation":all_abl,"comparison":{}},f,indent=2)
    print(f"  Ablation sauvegardée dans {RESULTS_FILE}");sys.stdout.flush()

# ============================================================
# PARTIE 2 : COMPARISON (6 méthodes × N × 3 seeds)
# ============================================================
if not comp_res or not any(comp_res.get(m,{}).get("c") for m in comp_res):
    print("\n=== COMPARISON ===");sys.stdout.flush()

    bp=TextBaselineProvider(mdl,tok,"mean_embedding");bv=bp.get_baseline_vector(device=device)

    methods=["dixai","random","integrated_grads","attention","lime","shap"]
    comp_res={m:{"s":[],"c":[]} for m in methods}

    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed)
        exs=load_sst2(limit=N,split="validation",seed=seed)
        ex_dixai=TextDecisionInformationExplainer(mdl,tok,lambda_fidelity=30.0,lambda_contiguity=2.0,baseline_type="mean_embedding",device=device)

        for i,e in enumerate(exs):
            if i%10==0:print(f"  seed={seed} [{i+1}/{N}]");sys.stdout.flush()

            # DIxAI
            ep=ex_dixai.explain(e.text,steps=STEPS,lr=0.1,init_logits=-1.0,seed=seed)
            tgt=ep.predicted_class;ids,am,cm=get_cm(tok,e.text,device);T=ids.shape[1]
            mk=max(1,int(((ep.mask_probs>0.5).float()*cm).sum().item()))
            er=compute_eraser_scores(mdl,tok,e.text,ep.mask_probs,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["dixai"]["s"].append(float(er.sufficiency));comp_res["dixai"]["c"].append(float(er.comprehensiveness))

            # Random
            rm=topk(torch.rand(T),cm,mk)
            er=compute_eraser_scores(mdl,tok,e.text,rm,bv,device=device,strategy="threshold",threshold=0.5)
            comp_res["random"]["s"].append(float(er.sufficiency));comp_res["random"]["c"].append(float(er.comprehensiveness))

            # IG
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
            except:pass

            # Attention
            try:
                with torch.no_grad():out=mdl(input_ids=ids,attention_mask=am,output_attentions=True)
                atts=getattr(out,"attentions",None)
                if atts:
                    cls_row=atts[-1][0].mean(0)[0].detach();am2=topk(cls_row,cm,mk)
                    er=compute_eraser_scores(mdl,tok,e.text,am2,bv,device=device,strategy="threshold",threshold=0.5)
                    comp_res["attention"]["s"].append(float(er.sufficiency));comp_res["attention"]["c"].append(float(er.comprehensiveness))
            except:pass

            # LIME
            try:
                from lime.lime_text import LimeTextExplainer
                def pf(texts):
                    enc=tok(texts,return_tensors="pt",truncation=True,padding=True)
                    enc={k:v.to(device) for k,v in enc.items()}
                    with torch.no_grad():out=mdl(**enc);probs=torch.softmax(out.logits,dim=-1)
                    return probs.cpu().numpy()
                le=LimeTextExplainer(class_names=["0","1"]).explain_instance(e.text,pf,num_features=50,num_samples=500,labels=[tgt])
                ws=dict(le.as_list(label=tgt))
                ls=torch.zeros(T);tokens=tok.convert_ids_to_tokens(ids[0].tolist())
                for ti,ts in enumerate(tokens):
                    cl=ts.replace("##","").lower().strip()
                    for w,sc in ws.items():
                        if cl and (cl in w.lower() or w.lower() in cl):ls[ti]=max(ls[ti].item(),abs(sc))
                lm=topk(ls,cm,mk)
                er=compute_eraser_scores(mdl,tok,e.text,lm,bv,device=device,strategy="threshold",threshold=0.5)
                comp_res["lime"]["s"].append(float(er.sufficiency));comp_res["lime"]["c"].append(float(er.comprehensiveness))
            except:pass

            # SHAP
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
                sv=shap_ex([e.text],max_evals=200,batch_size=8)
                vals=np.abs(sv.values[0,:,tgt])
                enc2=tok(e.text,return_tensors="pt",truncation=True);T2=enc2["input_ids"].shape[1]
                ss_scores=torch.zeros(T2)
                if len(vals)>=T2:ss_scores=torch.tensor(vals[:T2],dtype=torch.float32)
                else:ss_scores[:len(vals)]=torch.tensor(vals,dtype=torch.float32)
                sm=topk(ss_scores,cm,mk)
                er=compute_eraser_scores(mdl,tok,e.text,sm,bv,device=device,strategy="threshold",threshold=0.5)
                comp_res["shap"]["s"].append(float(er.sufficiency));comp_res["shap"]["c"].append(float(er.comprehensiveness))
            except:pass

    # Sauvegarder après comparison
    with open(RESULTS_FILE,"w") as f:
        json.dump({"ablation":all_abl,"comparison":comp_res},f,indent=2)
    print(f"\n  Comparison sauvegardée dans {RESULTS_FILE}");sys.stdout.flush()

# ============================================================
# PARTIE 3 : QUALITATIF (top 5 exemples)
# ============================================================
print("\n=== QUALITATIF ===");sys.stdout.flush()

qual_dir=os.path.join(RESULTS_DIR,"qualitative")
os.makedirs(qual_dir,exist_ok=True)

qual_data=[]
ex_dixai=TextDecisionInformationExplainer(mdl,tok,lambda_fidelity=30.0,lambda_contiguity=2.0,baseline_type="mean_embedding",device=device)
exs=load_sst2(limit=5,split="validation",seed=0)

for i,e in enumerate(exs):
    ep=ex_dixai.explain(e.text,steps=STEPS,lr=0.1,init_logits=-1.0,seed=0)
    tgt=ep.predicted_class
    selected=[t for t in ep.top_tokens(0.5) if t not in ["[CLS]","[SEP]","[PAD]"]]
    qual_data.append({
        "text":e.text,
        "predicted_class":int(tgt),
        "fidelity":float(ep.fidelity_score),
        "sparsity":float(ep.sparsity_score),
        "contiguity":float(ep.contiguity_score),
        "selected_tokens":selected,
        "mask_probs":{tok:float(p) for tok,p in zip(ep.tokens,ep.mask_probs.tolist())}
    })
    print(f"  [{i+1}/5] {e.text[:50]}... → {selected}");sys.stdout.flush()

with open(os.path.join(qual_dir,"qualitative.json"),"w") as f:
    json.dump(qual_data,f,indent=2,ensure_ascii=False)

# ============================================================
# PARTIE 4 : FIGURES
# ============================================================
print("\n=== FIGURES ===");sys.stdout.flush()

fig_dir=os.path.join(RESULTS_DIR,"figures")
os.makedirs(fig_dir,exist_ok=True)

# Figure 1 : Comparison barplot Comp
fig,ax=plt.subplots(figsize=(10,5))
method_labels=["Random","Attention","LIME","SHAP","IG","DIxAI-Text"]
method_keys=["random","attention","lime","shap","integrated_grads","dixai"]
comp_means=[np.mean(comp_res.get(m,{}).get("c",[0])) for m in method_keys]
comp_stds=[np.std(comp_res.get(m,{}).get("c",[0])) for m in method_keys]
suff_means=[np.mean(comp_res.get(m,{}).get("s",[1])) for m in method_keys]
suff_stds=[np.std(comp_res.get(m,{}).get("s",[1])) for m in method_keys]

x=np.arange(len(method_labels))
width=0.35
bars1=ax.bar(x-width/2,comp_means,width,label="Comprehensiveness↑",color="#2ecc71",alpha=0.8,yerr=comp_stds,capsize=3)
bars2=ax.bar(x+width/2,suff_means,width,label="Sufficiency↓",color="#e74c3c",alpha=0.8,yerr=suff_stds,capsize=3)
ax.set_ylabel("Score")
ax.set_title("ERASER Faithfulness on SST-2 (N=50, 3 seeds)")
ax.set_xticks(x)
ax.set_xticklabels(method_labels,rotation=15,ha="right")
ax.legend()
ax.set_ylim(0,max(max(comp_means)+0.1,0.5))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir,"sst2_comparison.png"),dpi=150)
plt.close()
print(f"  Figure sauvée : {fig_dir}/sst2_comparison.png");sys.stdout.flush()

# Figure 2 : Ablation barplot
fig,ax=plt.subplots(figsize=(12,5))
abl_names=["mean_emb","mask_token","pad_token","No Contig","Strong Contig","No Anneal"]
abl_keys=["Full DIxAI-Text (mean_emb)","Baseline=mask_token","Baseline=pad_token","No Contiguity","Strong Contiguity","No Annealing"]
abl_comp=[np.mean([r["comp_m"] for r in all_abl if r["config"]==k]) for k in abl_keys]
abl_comp_s=[np.std([r["comp_m"] for r in all_abl if r["config"]==k]) for k in abl_keys]