# Dataset layout

The repository does not redistribute BABEL, HumanML3D, AMASS, or generated
motion features. Obtain the source datasets from their official projects and
comply with their licenses.

Both loaders use Guo-style motion features under
`datasets/motions/guoh3dfeats/`. A feature file must have the same `path`
stem referenced by the corresponding annotation item.

## BABEL

1. Download BABEL v1.0 and the required AMASS motions.
2. Put the raw files in `datasets/babel/babel_v1.0_release/`.
3. Run `python prepare/babel_aau.py` to create the unified annotations.
4. Generate the Guo motion features.

The released checkpoint trains on processed train plus `extra_train`, retains
mirrored features, and evaluates on processed validation plus `extra_val`.
Within this codebase, BABEL's split name `test` denotes that processed
validation protocol; it is not an independent blind test.

## HumanML3D

Create `datasets/humanml3d/humanml3d.json`. Each top-level motion item must
contain `path`, `duration`, and an `annotations` list whose entries contain
`text`, `start`, and `end`. Put official motion IDs, one per line, in
`datasets/humanml3d/splits/train.txt`, `val.txt`, and `test.txt`, then generate
the matching Guo motion features.

The released HumanML3D checkpoint uses the standard train split without data
augmentation. Its reported diagnostic uses the complete official test split.

## Expected tree

```text
datasets/
├── babel/
│   ├── babel.json
│   ├── babel_extra.json
│   ├── babel_v1.0_release/
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
│   │   ├── extra_train.json
│   │   └── extra_val.json
│   └── splits/
│       ├── train.txt
│       ├── train_extra.txt
│       ├── test.txt
│       └── test_extra.txt
├── humanml3d/
│   ├── humanml3d.json
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
└── motions/
    └── guoh3dfeats/
```

During supervised training, missing query-agnostic segment priors are
generated in each dataset's cache directory. Prediction-only evaluation does
not read or generate those annotation-derived files.
