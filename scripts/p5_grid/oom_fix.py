#!/usr/bin/env python3
"""oom_fix.py RESULT.json -> prints how many problems it removed (0 if none or no file).
Cuts every (method, K) block back to the problem before its first error (error set or 0 tokens),
marks it partial so benchmark.py --resume reruns from there, backs the original up in backup_oom/."""
import json,os,sys,shutil,time
f=sys.argv[1]
if not os.path.exists(f): print(0); sys.exit()
d=json.load(open(f)); cut=0
for b in d.get("results",{}).values():
    for v in b.values():
        pp=v.get("per_problem",[]); i=next((j for j,p in enumerate(pp) if p.get("error") or p.get("n_tokens_generated",0)==0),None)
        if i is None: continue
        cut+=len(pp)-i; v.clear(); v.update({"partial":True,"n_total":len(pp),"per_problem":pp[:i]})
if cut:
    bk=os.path.join("/opt/dlami/nvme/kvcache/backup_oom",f"{os.path.basename(f)[:-5]}.{int(time.time())}.json"); shutil.copy2(f,bk)
    json.dump(d,open(f+".oomfix.tmp","w"),indent=2); os.replace(f+".oomfix.tmp",f)
print(cut)
