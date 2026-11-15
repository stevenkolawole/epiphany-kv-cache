#!/bin/bash
# run.sh <venv: rkv|main|kvpress> <tag> <baseline_costs.py args...>; card 7 only, refuses if the card is in use.
venv=$1; tag=$2; shift 2
case $venv in rkv) PY=/opt/dlami/nvme/kvcache_rkv/venv/bin/python;; main) PY=/opt/dlami/nvme/kvcache/venv/bin/python;; kvpress) PY=/opt/dlami/nvme/kvcache_cost/venv_kvpress/bin/python;; esac
used=$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 500 ]; then echo "$tag REFUSED card7 used=${used}MiB" >> ~/rebuttal_baseline_costs/status.txt; exit 3; fi
export HF_HOME=/opt/dlami/nvme/kvcache/hf USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1
mkdir -p ~/rebuttal_baseline_costs/logs
cd /opt/dlami/nvme/kvcache_cost/kvcache
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 9)" || { echo "$tag NOCUDA" >> ~/rebuttal_baseline_costs/status.txt; exit 9; }
echo "$tag START $(date +%T) $*" >> ~/rebuttal_baseline_costs/status.txt
$PY scripts/baseline_costs.py --output ~/rebuttal_baseline_costs/$tag.json "$@" > ~/rebuttal_baseline_costs/logs/$tag.log 2>&1
echo "$tag END rc=$? $(date +%T)" >> ~/rebuttal_baseline_costs/status.txt
