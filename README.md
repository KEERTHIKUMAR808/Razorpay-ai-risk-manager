# Dataset

## Recommended final evaluation: IEEE-CIS Fraud Detection

For the strongest Track 02 submission, use the public IEEE-CIS Fraud Detection dataset.

Download:
- `train_transaction.csv`
- `train_identity.csv`

Place both files in this folder:

```text
data/
├── train_transaction.csv
└── train_identity.csv
```

Do not commit the dataset to GitHub.

The training pipeline performs a chronological held-out split, so the evaluation set represents later transactions than the training set.

## Reproducible demo

If you do not have the dataset yet, run:

```bash
python train.py --demo
```

The demo dataset is clearly labeled as synthetic and is useful only for testing the software pipeline. Do **not** present demo metrics as real-world fraud performance in the Razorpay submission.
