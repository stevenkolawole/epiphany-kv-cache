# p5 grid runner (copy of what runs on uk-p5 under /opt/dlami/nvme/kvcache, Oct 28)

- `runner4.sh <card> <slot> <queue>`: one queue runner. Admission by free memory (line mem + 12 GB),
  pause hooks, OOM handling (stop at the first OOM in the log, `oom_fix.py` cuts the result back to the
  problem before the first error, requeue at the head with +24 GB, up to 4 attempts).
- `p5_keepalive.sh` (cron, every 10 min): MPS daemon, offload guard, reaper, breaker (genuine failures
  only), respawn of 6 slots per card, mirror to `$HOME`.
- `pause.sh soft|hard [card ...]`, `resume.sh [card ...]`.
- `validate_lines.py`: checks every queue line against the grid line it was derived from.
- `queue_oct28.txt`: the 3,416 lines queued on Oct 28 (gpqa_qwen3 and Nemotron reruns on H200;
  H2O, R-KV, LongFlow, ThinKV and EpiKV^-Q for Qwen3-14B, Qwen3-8B and Nemotron).
