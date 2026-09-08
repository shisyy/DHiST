# HumanML3D checkpoint release

HumanML3D is released for checkpoint inference and evaluation only. Use the
current [DHiST source](https://github.com/shisyy/DHiST) with the commands below.
HumanML3D training scripts and a training recipe are not provided.

```bash
python scripts/download_checkpoint.py --dataset humanml3d
python eval_predonly.py --dataset humanml3d --output outputs/humanml3d_49_0_eval
```

Prediction-only evaluation on the complete official test split (3,280 queries,
550 motions):

- R1@0.1/0.3/0.5/0.7/0.9: 74.4207 / 63.7500 / 51.6768 / 41.1280 / 15.5793
- R5@0.1/0.3/0.5/0.7/0.9: 75.7927 / 65.2134 / 53.0183 / 43.0793 / 20.7927
- mIoU / R5-mIoU: 49.0722 / 51.1590

The compact checkpoint contains all trained DHiST weights, LoRA tensors, and
the trained text projection. The unchanged frozen `roberta-base` backbone is
reconstructed at loading. The asset is 45,436,766 bytes. The reconstructed
tensor state was audited to match the original full checkpoint exactly.

Asset: `dhist_humanml3d_49_0_seed123_e15.ckpt.pkl`

SHA-256: `ba4fd38e3498547fdf13063539a6be25a060ddd8583e7a41c5aa872b034991c7`

The checkpoint's original epoch and iteration metadata are retained. The
evaluation configuration contains model-loading settings and no optimizer or
training schedule. Shared criterion-construction fields remain for checkpoint
compatibility; evaluation does not compute a training loss.

All six label-like network inputs are forced to zero. Label perturbation
audits report a maximum prediction-output difference of 0.0. Ground-truth
timestamps are used by the metric accumulator after prediction. This result
is a previously exposed official-test diagnostic, not a new blind or
independent test score. It is distinct from the paper row
50.80 / 41.58 / 16.79 / 49.07.
