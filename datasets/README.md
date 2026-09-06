# Dataset layout

The repository does not redistribute BABEL, AMASS, or generated motion features.

During supervised training, missing query-agnostic segment-prior files are
generated under `datasets/babel/preprocess/segment_priors/`. Prediction-only
evaluation does not read or generate those annotation-derived files.

1. Download BABEL v1.0 from the official BABEL website.
2. Download the required AMASS motion data and create HumanML3D/Guo motion features.
3. Place the raw BABEL files in `datasets/babel/babel_v1.0_release/`.
4. Run `python prepare/babel_aau.py` to create the unified annotations.
5. Place the derived features in `datasets/motions/guoh3dfeats/`.

Expected layout:

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
└── motions/
    └── guoh3dfeats/
```

Within this codebase, the split name `test` denotes the processed BABEL validation protocol. It must not be interpreted as an independent blind-test result.
