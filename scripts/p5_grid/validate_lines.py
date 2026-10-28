#!/usr/bin/env python3
"""Check every new queue line against the grid line it was derived from. Prints one verdict per check."""
import re,sys,collections
S=sys.argv[1]; man_f=sys.argv[2]
man={l.split()[3]:l.rstrip("\n") for l in open(man_f) if len(l.split())>=5}
files=["q_rerun.txt","q_math.txt","q_gpqa.txt","q_noq.txt","q_nemo.txt","q_nemo_grid.txt"]
Q=["--query_sim_mode","--query_mode","--query_sim_window","--query_sim_weight"]
BANDS={"q14b":("12","26"),"qwen3":("11","24"),"nemo":(None,None)}
MODEL={"q14b":"Qwen/Qwen3-14B","qwen3":"Qwen/Qwen3-8B","nemo":"nvidia/Llama-3.1-Nemotron-Nano-8B-v1"}
METH={"h2o":"h2o","r_kv":"r_kv","longflow":"longflow","thinkv_faithful":"thinkv_faithful","epikvnoq":"kv_seg_hs",
      "epikv":"kv_seg_hs_fill","seg":"kv_seg_hs","raas":"raas","none":"none","lagkv":"lag_kv"}
fail=collections.Counter(); tags=collections.Counter(); outs=collections.Counter(); n=0
def arg(L,k):
    m=re.search(rf"--{k} (\S+)",L); return m.group(1) if m else None
for fn in files:
    for L in open(f"{S}/{fn}"):
        L=L.rstrip("\n")
        if not L.strip(): continue
        n+=1; f=L.split(" ",4); tag=f[3]; tags[tag]+=1; out=arg(L,"output"); outs[out]+=1
        m=re.match(r"(?:h200_)?(gpqa|math)_(q14b|qwen3|nemo)_(\w+?)_K(\d+)_s(\d+)$",tag)
        if not m: fail["tag-format"]+=1; continue
        ds,model,meth,K,s=m.groups()
        def chk(c,name):
            if not c: fail[name]+=1
        chk(arg(L,"model")==MODEL[model],"model"); chk(arg(L,"dataset")==("gpqa" if ds=="gpqa" else "math500"),"dataset")
        chk(arg(L,"methods")==METH[meth],"method"); chk(arg(L,"cache_sizes")==K,"budget")
        chk(arg(L,"max_new_tokens")=="16384","cap16384"); chk("--logical_positions" in L,"logical"); chk(arg(L,"keep_recent_k")=="128","recent128")
        chk(arg(L,"start_idx")==str(int(s)),"start_idx"); chk(arg(L,"n_samples")==("11" if ds=="gpqa" else "20"),"n_samples")
        chk(("--system_prompt" in L)==(model=="nemo"),"sysprompt")
        hasq=any(q in L for q in Q)
        chk(hasq==(meth=="epikv" or (meth=="seg" and tag.startswith("h200_"))),"query-flags")
        if meth in ("epikvnoq","epikv","seg"):
            a,b=BANDS[model]; chk(arg(L,"band_a_layer")==a and arg(L,"band_b_layer")==b,"bands")
            chk(arg(L,"refresh_tau")=="128" and arg(L,"tau_mode")=="steps","tau128steps")
        chk(out and out.endswith(f"/{tag}.json") and any(f"/{x}/" in out for x in ("results_baselines","results_gpqa_qwen3_h200","results_nemo_h200")),"output-path")
        chk(f"mkdir -p {out.rsplit('/',1)[0]} " in L,"mkdir")
        chk(f[1].isdigit() and int(f[1])>=(40 if model=="q14b" else 30),"mem")
        chk(tag not in man,"tag-collides-with-grid")
        # same command as the grid line for this cell apart from the fields that are meant to differ
        base={"epikvnoq":"seg"}.get(meth, "raas" if meth in ("h2o","r_kv","longflow","thinkv_faithful") else meth)
        bs=int(s) if ds=="gpqa" else (int(s)//100)*100
        bm="q14b" if (ds=="math" and model=="qwen3") else model
        bt=f"{ds}_{bm}_{base}_K{K}_s{bs}"
        B=man.get(bt)
        if not B: fail["no-template"]+=1; continue
        norm=lambda x: " ".join(re.sub(r"--(output|start_idx|n_samples|methods|band_a_layer|band_b_layer|model) \S+|mkdir -p \S+|"+"|".join(q+r" \S+" for q in Q),"",x.split(" ",4)[4]).split())
        chk(norm(L)==norm(B),"differs-from-template")
dup_t=sum(1 for v in tags.values() if v>1); dup_o=sum(1 for v in outs.values() if v>1)
print(f"lines {n}  duplicate tags {dup_t}  duplicate outputs {dup_o}")
print("failures:", dict(fail) if fail else "none")
