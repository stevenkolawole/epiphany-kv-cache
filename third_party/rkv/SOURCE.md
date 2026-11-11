Vendored unmodified from https://github.com/Zefan-Cai/R-KV (Apache-2.0, see LICENSE), directory HuggingFace/rkv,
commit a329f2950b5500f245b6e0471cc0f4525a91287c (2026-11-09). Needs transformers>=4.48.1,<4.56 (its requirements.txt:
newer versions produce garbled output); run it only in a venv inside that range (p4d: /opt/dlami/nvme/kvcache_rkv/venv,
transformers 4.55.4). Hyperparameters passed by benchmark.py are the defaults of HuggingFace/run_math.py: window_size 8,
mix_lambda 0.1, retain_ratio 0.2, retain_direction last, divide_method step_length, divide_length 128, update_kv True.
