#!/usr/bin/env bash
# E14 GPU queue (16 #40): wait until the GPU is free, then E12 re-evaluation (per-video CIs + detection cache for
# compare_runs.py), then E14 training. Written after the first E14 start crawled because another project's local LLM
# (Ollama llama-server, ~5 GB VRAM, ~18 GB commit) was sharing the 8 GB GPU and the paging file.
#   bash experiments/run_e14_queue.sh        (from the repo root, Git Bash)
set -u
PY=/c/cuas_venv/Scripts/python.exe
LOG=/c/cuas_data/runs

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "$(date '+%F %T') waiting for a free GPU (used: $(gpu_used) MiB)"
while [ "$(gpu_used)" -gt 1500 ]; do sleep 60; done
echo "$(date '+%F %T') GPU free ($(gpu_used) MiB used): E12 re-evaluation"
$PY experiments/e12_eval.py E12_multidata_yolo26s > "$LOG/eval_E12_multidata_yolo26s_v2.txt" 2>&1
echo "$(date '+%F %T') E12 re-evaluation exit $?; starting E14 training"
$PY experiments/train_local.py --data C:/cuas_data/e14/data_e14.yaml --name E14_hardneg_yolo26s > "$LOG/E14_hardneg_yolo26s.log" 2>&1
echo "$(date '+%F %T') E14 training exit $?"
