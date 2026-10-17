#!/usr/bin/env python3
"""Merge split pieces back into their cell file (Oct 8). See plan_splits.py.

A cell merges only when its prefix file is unchanged since planning and EVERY piece is
complete (partial false), error-free, and its problem texts match the harness's own
load_problems() for that range. The merged block is exactly what one uninterrupted run
writes (same keys, accuracy/n_correct/means recomputed from per_problem). The pre-merge
file is kept in splits/premerge/; splits/provenance/<tag>.json records which box ran each
problem index, for the A100-vs-H200 consistency check. Writes splits/active_tags.txt
(planned, not yet merged) for the status block. Idempotent; run from cron.
usage: merge_splits.py [--dry]"""
import json,os,sys,shutil,time
K="/opt/dlami/nvme/kvcache"; SP=K+"/splits"; dry="--dry" in sys.argv
plan=json.load(open(SP+"/plan.json"))
os.makedirs(SP+"/merged",exist_ok=True); os.makedirs(SP+"/premerge",exist_ok=True); os.makedirs(SP+"/provenance",exist_ok=True)
lp=None
def problems(s,n):
    global lp
    if lp is None:
        sys.path.insert(0,K+"/kvcache/scripts"); os.environ.setdefault("HF_HOME",K+"/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE","1")
        import benchmark; lp=benchmark.load_problems
    return lp("math500",n,s)
bad=lambda p: bool(p.get("error")) or p.get("n_tokens_generated",0)==0
active=[]; log=[]
for t,v in plan.items():
    if os.path.exists(f"{SP}/merged/{t}"): continue
    active.append(t)
    pieces=[]; ok=True; why=""
    for p in v["pieces"]:
        f=p["out"]
        if not os.path.exists(f): ok=False; why=f"missing {p['tag']}"; break
        d=json.load(open(f)); bl=[b for m in d["results"].values() for b in m.values()]
        if len(bl)!=1 or bl[0].get("partial") or "accuracy" not in bl[0]: ok=False; why=f"incomplete {p['tag']}"; break
        pp=bl[0]["per_problem"]
        if any(bad(x) for x in pp): ok=False; why=f"ERRORED {p['tag']}"; break
        pieces.append((p,pp))
    if not ok:
        if why.startswith("ERRORED"): log.append(f"{t}: {why}")
        continue
    main=json.load(open(v["main"])); mb=[(m,k,b) for m,bb in main["results"].items() for k,b in bb.items()]
    assert len(mb)==1; m,k,blk=mb[0]; pre=blk["per_problem"]
    if len(pre)!=v["prefix"]: log.append(f"{t}: prefix changed {v['prefix']}->{len(pre)}, not merged"); continue
    allp=list(pre); prov={i:"p5" for i in range(len(pre))}
    for p,pp in pieces:
        for j,x in enumerate(pp): prov[p["a"]+j]=p["box"]
        allp+=pp
    exp=problems(v["start_idx"],v["n_samples"])
    if len(exp)!=len(allp) or any(e["problem"][:80]!=x["problem"][:80] for e,x in zip(exp,allp)):
        log.append(f"{t}: PROBLEM MISMATCH (expected {len(exp)}, have {len(allp)}), not merged"); continue
    times=[x["wall_time_s"] for x in allp if x.get("wall_time_s")]; mems=[x["peak_gpu_mb"] for x in allp if x.get("peak_gpu_mb")]
    nc=sum(1 for x in allp if x.get("correct"))
    newb={"accuracy":nc/len(allp),"n_correct":nc,"n_total":len(allp),
          "mean_wall_time_s":round(sum(times)/len(times),2) if times else None,
          "mean_peak_gpu_mb":round(sum(mems)/len(mems),1) if mems else None,"per_problem":allp}
    main["results"][m][k]=newb; main["meta"]["n_samples"]=len(allp)
    if dry: log.append(f"{t}: READY {len(allp)} problems acc {nc}/{len(allp)}"); continue
    shutil.copy2(v["main"],f"{SP}/premerge/{t}.json")
    json.dump({"tag":t,"box_by_index":prov,"p4d_indices":sorted(i for i,b in prov.items() if b=="p4d")},open(f"{SP}/provenance/{t}.json","w"))
    tmp=v["main"]+".merge.tmp"; json.dump(main,open(tmp,"w"),indent=2); os.replace(tmp,v["main"])
    open(f"{SP}/merged/{t}","w").write(time.strftime("%FT%TZ",time.gmtime())+"\n"); active.remove(t)
    log.append(f"{t}: MERGED {len(allp)} problems, acc {nc}/{len(allp)}, p4d problems {sum(b=='p4d' for b in prov.values())}")
if not dry: open(SP+"/active_tags.txt","w").write("\n".join(active)+("\n" if active else ""))
ts=time.strftime("%FT%TZ",time.gmtime())
with open(SP+"/merge.log","a") as L:
    for l in log: L.write(f"{ts} {l}\n")
print(f"{ts} active {len(active)}", *log, sep="\n")
