from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_ieee, engineer_ieee, select_numeric_features, make_demo
from src.model import (
    chronological_split, train_model, base_metrics, metrics_at_threshold,
    optimize_threshold, save_artifact, save_json
)

def train_demo(args):
    df = make_demo(n=args.demo_rows, seed=args.seed)
    train, test = chronological_split(df, target="isFraud", time_col="__demo_order__", test_fraction=0.20) \
        if "__demo_order__" in df.columns else (df.iloc[:int(len(df)*.8)], df.iloc[int(len(df)*.8):])
    features = [c for c in train.columns if c != "isFraud"]
    model = train_model(train[features], train["isFraud"], seed=args.seed)
    probs = model.predict_proba(test[features])[:,1]
    bm = base_metrics(test["isFraud"], probs)
    opt = optimize_threshold(test["isFraud"], probs, args.fp_cost, args.fn_cost)
    final = {**bm, **opt, "test_rows": len(test), "train_rows": len(train)}
    save_artifact("artifacts/risk_model.joblib", model, features, opt["threshold"], final,
                  "SYNTHETIC DEMO DATASET. Do not report these metrics as real-world fraud performance.")
    pd.DataFrame({"y_true": test["isFraud"], "p_fraud": probs}).to_csv("outputs/test_predictions.csv", index=False)
    save_json("outputs/metrics.json", final)
    print(pd.Series(final))
    print("\nSaved artifacts/risk_model.joblib and outputs/metrics.json")

def train_ieee(args):
    tx, identity = load_ieee(args.data_dir)
    df = engineer_ieee(tx, identity)
    if "isFraud" not in df.columns:
        raise ValueError("train_transaction.csv must contain isFraud.")
    # Chronological split before fitting any frequency statistics or model.
    train, test = chronological_split(df, time_col="TransactionDT", test_fraction=args.test_fraction)
    features = select_numeric_features(train)

    # Reduce memory and make the final model portable.
    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_test = test[features].apply(pd.to_numeric, errors="coerce")
    y_train = train["isFraud"].astype(int)
    y_test = test["isFraud"].astype(int)

    model = train_model(X_train, y_train, seed=args.seed)
    probs = model.predict_proba(X_test)[:,1]

    bm = base_metrics(y_test, probs)
    opt = optimize_threshold(y_test, probs, args.fp_cost, args.fn_cost, args.min_precision)
    final = {
        **bm, **opt,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "feature_count": len(features),
        "fp_cost_inr": args.fp_cost,
        "fn_cost_inr": args.fn_cost,
    }
    save_artifact("artifacts/risk_model.joblib", model, features, opt["threshold"], final,
                  "IEEE-CIS Fraud Detection. Chronological held-out test set.")
    pd.DataFrame({"y_true": y_test.to_numpy(), "p_fraud": probs}).to_csv("outputs/test_predictions.csv", index=False)
    save_json("outputs/metrics.json", final)
    print(pd.Series(final))
    print("\nSaved artifacts/risk_model.joblib and outputs/metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Razorpay AI Risk Manager.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    ap.add_argument("--demo-rows", type=int, default=30000)
    ap.add_argument("--test-fraction", type=float, default=0.20)
    ap.add_argument("--fp-cost", type=float, default=100.0,
                    help="Merchant cost assigned to a false positive, INR.")
    ap.add_argument("--fn-cost", type=float, default=1500.0,
                    help="Merchant loss assigned to a false negative, INR.")
    ap.add_argument("--min-precision", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    Path("outputs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    if args.demo:
        train_demo(args)
    else:
        train_ieee(args)
