#!/usr/bin/env python3
"""Plan 10-problem pieces for long math_q14b cells (Oct 8). For each tag: the cell's own
file keeps its finished prefix [0,n); problems [n,100) become pieces, each its own job with
--start_idx s+a --n_samples len, writing splits/<tag>__p<aaa>.json next to the cell file.
merge_splits.py appends the pieces back in order once all are complete and clean."""
import json,glob,os,re,sys
K="/opt/dlami/nvme/kvcache"; CH=10
tags=open(sys.argv[1]).read().split()
man={l.split()[3]:l.rstrip("\n") for l in open(K+"/epikv_master.txt") if len(l.split())>=5}
plan={}; lines=[]
for t in tags:
    L=man[t]; out=re.search(r"--output (\S+)",L).group(1); s=int(re.search(r"--start_idx (\d+)",L).group(1))
    N=int(re.search(r"--n_samples (\d+)",L).group(1))
    n=0
    if os.path.exists(out):
        d=json.load(open(out)); blocks=[v for b in d["results"].values() for v in b.values()]
        assert len(blocks)==1, t
        pp=blocks[0]["per_problem"]; n=len(pp)
        assert not any(p.get("error") or p.get("n_tokens_generated",0)==0 for p in pp), ("errored prefix",t)
    d_out=os.path.dirname(out); pieces=[]; a=n
    while a<N:
        ln=min(CH,N-a)
        if N-(a+ln)<5: ln=N-a
        pt=f"{t}__p{a:03d}"; po=f"{d_out}/splits/{pt}.json"
        pl=L.replace(f" {t} ",f" {pt} ",1).replace(f"--start_idx {s} ",f"--start_idx {s+a} ").replace(f"--n_samples {N} ",f"--n_samples {ln} ").replace(f"--output {out}",f"--output {po}").replace(f"mkdir -p {d_out} ",f"mkdir -p {d_out}/splits ")
        assert pl.count(f"--start_idx {s+a} ")==1 and pl.count(f"--n_samples {ln} ")==1 and pl.count(po)==1 and f"{d_out}/splits " in pl, pt
        pieces.append({"tag":pt,"a":a,"len":ln,"out":po}); lines.append(pl); a+=ln
    plan[t]={"main":out,"start_idx":s,"n_samples":N,"prefix":n,"pieces":pieces}
os.makedirs(K+"/splits",exist_ok=True)
json.dump(plan,open(K+"/splits/plan.json","w"),indent=1)
open(K+"/splits/piece_lines.txt","w").write("\n".join(lines)+"\n")
print("cells",len(plan),"pieces",len(lines),"problems",sum(p["len"] for v in plan.values() for p in v["pieces"]))
