# Pair each window-rule, kvpress and native-LongFlow config with EpiKV (kv_seg_hs_fill) at the same model and K, on
# MATH-500 problems both have finished. Prints accuracy, paired wins, exact McNemar p and cap hits (a run that
# reaches the token cap without answering; the paper's loop count). A row is final only when pairs = 495.
import json, glob, re, collections, sys
from math import comb
WR, LF, CAP = sys.argv[1], sys.argv[2], 16384
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
cfg = collections.defaultdict(dict)
for f in glob.glob(f'{WR}/cells/*.json'):
    m = re.match(r'(.+_K\d+)_s(\d+)$', f.split('/')[-1][:-5])
    if not m: continue
    s = int(m.group(2))
    for v in json.load(open(f))['results'].values():
        for b in v.values():
            for i, x in enumerate(b['per_problem']):
                if not x.get('error') and x.get('n_tokens_generated'):
                    cfg[m.group(1)][s + i] = (bool(x['correct']), x['n_tokens_generated'] >= CAP)
for f in glob.glob(f'{LF}/results/*.json'):
    K = re.search(r'_K(\d+)_s', f).group(1)
    for b in json.load(open(f))['results']['longflow_native'].values():
        for x in b['per_problem']:
            if x.get('n_tokens_generated'):
                cfg[f'qwen_longflow_native_K{K}'][x['problem_idx']] = (bool(x['correct']), x['n_tokens_generated'] >= CAP)
print(f"{'config':38s} {'pairs':>5s} {'meth%':>6s} {'epikv%':>6s} {'only_m':>6s} {'only_e':>6s} {'p':>7s} {'cap_m':>5s} {'cap_e':>5s} {'oe_cap':>6s}")
for k in sorted(cfg):
    if 'kv_seg_hs_fill' in k: continue
    e = cfg.get(re.sub(r'(kvpress_snapkv_decoding|raas_epikv_window|r_kv_epikv_window|longflow_native)', 'kv_seg_hs_fill', k), {})
    idx = sorted(set(cfg[k]) & set(e)); n = len(idx)
    if not n: continue
    M, E = [cfg[k][i] for i in idx], [e[i] for i in idx]
    om = sum(m[0] and not x[0] for m, x in zip(M, E)); oe = sum(x[0] and not m[0] for m, x in zip(M, E))
    oe_cap = sum(x[0] and not m[0] and m[1] for m, x in zip(M, E))
    print(f"{k:38s} {n:5d} {100*sum(m[0] for m in M)/n:6.1f} {100*sum(x[0] for x in E)/n:6.1f} {om:6d} {oe:6d} "
          f"{mcnemar(om, oe):7.4f} {sum(m[1] for m in M):5d} {sum(x[1] for x in E):5d} {oe_cap:6d}")
