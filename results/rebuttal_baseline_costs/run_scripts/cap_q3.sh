export HF_HOME=/opt/dlami/nvme/kvcache/hf USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1
cd /opt/dlami/nvme/kvcache_cost/kvcache
PY=/opt/dlami/nvme/kvcache/venv/bin/python
echo "capacity_fresh_epikv_qwen3 START $(date +%T)" >> ~/rebuttal_baseline_costs/status.txt
for K in 512 1024 2048 4096; do
  $PY scripts/capacity_search.py --python $PY --K $K --max_batch 256 --output ~/rebuttal_baseline_costs/capacity_fresh_epikv_qwen3_K$K.json -- --method epikv --model Qwen/Qwen3-8B --band_a_layer 11 --band_b_layer 24 >> ~/rebuttal_baseline_costs/logs/capacity_fresh_epikv_qwen3.log 2>&1
done
echo "capacity_fresh_epikv_qwen3 END $(date +%T)" >> ~/rebuttal_baseline_costs/status.txt
