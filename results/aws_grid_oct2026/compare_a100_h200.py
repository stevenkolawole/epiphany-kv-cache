#!/usr/bin/env python3
"""A100 (p4d) vs H200 (p5) consistency check for the Oct 2026 AWS grid.

p4d ran 16 math_q14b split pieces and 16 whole gpqa_q14b cells. The same 32 jobs were
rerun on p5 (results_consistency/, tags cons_*), same code md5s, same libraries
(torch 2.6.0, transformers 4.57.1, flash-attn 2.7.4), same Qwen3-14B revision.
Greedy decoding, so any difference is hardware numerics. Per problem: identical answer,
identical correctness, identical token count (a proxy for an identical generation)."""
import json,glob,os,sys
R=os.path.dirname(os.path.abspath(__file__))
def blk(f): (m,b),=json.load(open(f))["results"].items(); (k,v),=b.items(); return v["per_problem"]
rows=[]
for f in sorted(glob.glob(f"{R}/p5/results_consistency/*.json")):
    t=os.path.basename(f)[:-5]
    a100=(glob.glob(f"{R}/p4d_epikv/results_*/splits/{t}.json")+glob.glob(f"{R}/p4d_epikv/results_*/{t}.json"))
    if not a100: print("no A100 file for",t); continue
    A=blk(a100[0]); H=blk(f)
    if len(A)!=len(H): print("length mismatch",t,len(A),len(H)); continue
    for i,(x,y) in enumerate(zip(A,H)):
        assert x["problem"][:60]==y["problem"][:60], (t,i)
        rows.append((t,i,x,y))
n=len(rows)
if not n: sys.exit("nothing to compare yet")
same_pred=sum(x["pred_answer"]==y["pred_answer"] for _,_,x,y in rows)
same_corr=sum(bool(x["correct"])==bool(y["correct"]) for _,_,x,y in rows)
same_tok=sum(x["n_tokens_generated"]==y["n_tokens_generated"] for _,_,x,y in rows)
a_acc=sum(bool(x["correct"]) for _,_,x,_ in rows); h_acc=sum(bool(y["correct"]) for _,_,_,y in rows)
a_only=sum(bool(x["correct"]) and not y["correct"] for _,_,x,y in rows); h_only=sum(bool(y["correct"]) and not x["correct"] for _,_,x,y in rows)
print(f"problems compared: {n}  (cells/pieces: {len(set(r[0] for r in rows))})")
print(f"identical token count (proxy for identical generation): {same_tok}/{n} = {same_tok/n:.1%}")
print(f"identical answer: {same_pred}/{n} = {same_pred/n:.1%}   identical correctness: {same_corr}/{n} = {same_corr/n:.1%}")
print(f"accuracy  A100 {a_acc}/{n} = {a_acc/n:.1%}   H200 {h_acc}/{n} = {h_acc/n:.1%}   discordant: A100-only right {a_only}, H200-only right {h_only}")
for g in ("math","gpqa"):
    rs=[r for r in rows if r[0].startswith(g)]
    if rs: print(f"  {g}: n={len(rs)} acc A100 {sum(bool(r[2]['correct']) for r in rs)/len(rs):.1%} H200 {sum(bool(r[3]['correct']) for r in rs)/len(rs):.1%} same-tokens {sum(r[2]['n_tokens_generated']==r[3]['n_tokens_generated'] for r in rs)/len(rs):.1%}")
