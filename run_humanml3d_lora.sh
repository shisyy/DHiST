#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:?Usage: bash run_humanml3d_lora.sh GPU_ID}
export CUDA_VISIBLE_DEVICES="${gpu_id}"

python train.py model=mlp_lora data=humanml3d \
  experiment=humanml3d_babel_aligned_lora \
  train_split=train eval_split=val seed=123 \
  machine.batch_size=64 machine.num_epochs=25 \
  model.optim.learning_rate=0.0002 model.optim.decay_step=50 \
  model.preserve_pretrained_text=true \
  model.alpha=0.8 model.beta=0.1 \
  model.boundary_smoothing_sigma=3.0 \
  model.kl_mask_strategy=valid \
  data.all_h_label_mode=sigmoid_count
