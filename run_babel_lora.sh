#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:?Usage: bash run_babel_lora.sh GPU_ID}
export CUDA_VISIBLE_DEVICES="${gpu_id}"

python train.py model=mlp_lora data=babel \
  seed=42 machine.batch_size=64 machine.num_epochs=100 \
  model.optim.learning_rate=0.0002 model.optim.decay_step=50 \
  model.alpha=0.8 model.beta=0.1
