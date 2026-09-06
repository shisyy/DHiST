# DHiST: BABEL release

Official BABEL implementation of **DHiST: Dual-Highlight Spatio-Temporal Learning for Temporal Sentence Grounding in 3D Human Motions**.

This repository is a clean source release. It contains code, configuration, split lists, the lightweight LoRA adapter, and audited result metadata. It intentionally excludes datasets, motion features, experiment logs, generated outputs, machine-specific paths, and development history.

## Released BABEL result

The released single checkpoint was evaluated on the BABEL validation protocol used by this project. Checkpoint selection and reporting therefore use the same validation set; this result is not an independent or blind test score.

| R1@0.1 | R1@0.3 | R1@0.5 | R1@0.7 | R1@0.9 | mIoU |
|---:|---:|---:|---:|---:|---:|
| 51.9542 | 45.2408 | 38.0839 | 29.6690 | 13.0040 | **35.4063** |

The full twelve-metric record is in [`artifacts/babel_35_3/summary.json`](artifacts/babel_35_3/summary.json).

### Checkpoint

The 45,428,296-byte compact checkpoint is attached to the [`babel-v1.0` release](https://github.com/shisyy/DHiST/releases/tag/babel-v1.0). It contains all DHiST weights, the 48 trained LoRA tensors, and the trained text projection. The unchanged frozen RoBERTa backbone is intentionally omitted and is loaded from `roberta-base` during initialization.

- Release asset: `dhist_babel_35_3_epoch94.ckpt.pkl`
- Checkpoint epoch field: `94`
- Training iteration field: `37479`
- Checkpoint format: `dhist-lora-compact-v1`
- SHA-256: `e014dcc8ec165bb9b0b2aa15d834e2fceddbf37c1fcd185c5de404e1bb233f83`

Download and verify it with:

```bash
python scripts/download_checkpoint.py
```

## Prediction-only inference

The public evaluation path uses only model predictions:

- all label-like tensors passed to the network are replaced with zeros;
- the public data path constructs those tensors as zeros and never loads
  annotation-derived segment-prior caches;
- label corruption is disabled;
- every module, including the LoRA-wrapped RoBERTa encoder, is forced into deterministic evaluation mode;
- the decoder uses predicted LP start/end logits and predicted highlight scores only;
- ground-truth timestamps are retained exclusively by the metric accumulator.

Changing the six label tensors between their original, zero, and randomized forms produced a maximum absolute difference of `0.0` for all six audited prediction outputs. See [`label_invariance_audit.json`](artifacts/babel_35_3/label_invariance_audit.json).

## Installation

The audited environment used Python 3.9.23, PyTorch 2.5.0+cu118, Transformers 4.38.0, and PEFT 0.10.0.

```bash
conda create -n dhist python=3.9 -y
conda activate dhist
# Install a PyTorch build appropriate for your CUDA version first.
pip install -r requirements.txt
```

Download the text encoders:

```bash
mkdir -p deps
git lfs install
git clone https://huggingface.co/roberta-base deps/roberta-base
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2 deps/all-mpnet-base-v2
```

## Data preparation

BABEL and AMASS data are not redistributed. Obtain them from their official sources and accept their licenses. Follow [`datasets/README.md`](datasets/README.md) to build the expected layout and motion features.

The project uses processed BABEL train plus `extra_train` for training and processed validation plus `extra_val` for validation. Mirrored motion features are retained, matching the released checkpoint protocol.

## Evaluation

After preparing the data and downloading the checkpoint:

```bash
python eval.py
```

Equivalent explicit command:

```bash
python eval_babel_predonly.py \
  --config artifacts/babel_35_3/config.yaml \
  --checkpoint artifacts/babel_35_3/ckpt.pkl \
  --output outputs/babel_35_3_eval
```

The evaluator writes `summary.json`, predictions, logs, and a fresh label-invariance audit to the output directory.

## Training

The released checkpoint used seed 42, 100 epochs, batch size 64, AdamW, learning rate `2e-4`, decay step 50, `alpha=0.8`, and `beta=0.1`:

```bash
python train.py model=mlp_lora data=babel \
  seed=42 machine.batch_size=64 machine.num_epochs=100 \
  model.optim.learning_rate=0.0002 model.optim.decay_step=50 \
  model.alpha=0.8 model.beta=0.1
```

Training is stochastic, so an independent run is not guaranteed to reproduce every decimal of the released checkpoint.

## License and acknowledgments

Code is released under the Apache License 2.0. This implementation builds on the open-source MLP codebase and includes components inspired by VSLNet, LGI, HumanML3D, TMR, and AMASS-Annotation-Unifier. Please follow the licenses and citation requirements of BABEL, AMASS, RoBERTa, MPNet, and the upstream projects.
