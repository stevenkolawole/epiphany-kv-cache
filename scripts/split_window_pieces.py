# Split window-rule pieces into C-problem pieces (default 25): the original line keeps its finished problems plus C more,
# new lines (tag _s<start>) cover the rest. The queue is reordered so the --defer tags run after the others.
# Run under the queue lock. Args: [--chunk C] --split TAG... --defer TAG...
import json, os, sys, re
W = '/home/skolawol/rebuttal_window_rules'; G = '/home/skolawol/gridq'
args = sys.argv[1:]; C = 25
if args[0] == '--chunk': C = int(args[1]); args = args[2:]
i = args.index('--defer'); split, defer = args[1:i], args[i + 1:]
L = open(f'{W}/lines.txt').read().rstrip('\n').split('\n')
num = {l.split()[0]: k + 1 for k, l in enumerate(L)}
new_first = []
for tag in split:
    k = num[tag]; line = L[k - 1]; S = int(re.search(r'--start_idx (\d+)', line).group(1))
    n_old = int(re.search(r'--n_samples (\d+)', line).group(1)); tot = min(n_old, 495 - S)
    f = f'{W}/cells/{tag}.json'; d = 0
    if os.path.exists(f):
        r = json.load(open(f))['results']; d = sum(len(b['per_problem']) for v in r.values() for b in v.values())
    keep = min(tot, d + C)
    L[k - 1] = line.replace(f'--n_samples {n_old}', f'--n_samples {keep}')
    for s in range(S + keep, S + tot, C):
        nt = re.sub(r'_s\d+$', f'_s{s}', tag); assert nt not in num, nt
        nl = line.replace(tag, nt, 1).replace(f'--start_idx {S}', f'--start_idx {s}').replace(f'--n_samples {n_old}', f'--n_samples {min(C, S + tot - s)}')
        L.append(nl); num[nt] = len(L); new_first.append(f'wr {len(L)}')
    print(tag, 'done', d, 'keep', keep, 'new', (tot - keep + C - 1) // C)
open(f'{W}/lines.txt.tmp', 'w').write('\n'.join(L) + '\n'); os.replace(f'{W}/lines.txt.tmp', f'{W}/lines.txt')
q = [l for l in open(f'{G}/queue.txt').read().split('\n') if l.strip()]
dset = {f'wr {num[t]}' for t in defer}
front = [l for l in q if l not in dset and (l.startswith('lf ') and int(l.split()[2]) <= 2048 or l.startswith('wr ') and not L[int(l.split()[1]) - 1].startswith('qwen_') or l.startswith('wr ') and L[int(l.split()[1]) - 1].startswith('qwen_kv_seg_hs_fill'))]
back = [l for l in q if l not in front]
moved = [l for l in back if l in dset]; rest = [l for l in back if l not in dset]
q2 = front + new_first + moved + rest
open(f'{G}/queue.txt.tmp', 'w').write('\n'.join(q2) + '\n'); os.replace(f'{G}/queue.txt.tmp', f'{G}/queue.txt')
print('queue', len(q), '->', len(q2), 'front', len(front), 'new', len(new_first), 'deferred', len(moved))
