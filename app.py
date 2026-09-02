from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.explain import reason_codes

st.set_page_config(page_title="Razorpay AI Risk Manager", page_icon="🛡️", layout="wide")

ARTIFACT = Path("artifacts/risk_model.joblib")

@st.cache_resource
def load_artifact():
    if not ARTIFACT.exists():
        return None
    return joblib.load(ARTIFACT)

artifact = load_artifact()

st.title("🛡️ Razorpay AI Risk Manager")
st.caption("Track 02 • Defense-only transaction risk scoring • Explainable decisions • Cost-aware thresholding")

if artifact is None:
    st.warning("No trained model found. Run `python train.py --demo` for a software demo, or train on IEEE-CIS before presenting final metrics.")
    st.stop()

model = artifact["model"]
features = artifact["features"]
default_threshold = float(artifact["threshold"])
saved_metrics = artifact.get("metrics", {})

tab1, tab2, tab3 = st.tabs(["Live Risk Verifier", "Held-out Evaluation", "Submission Evidence"])

with tab1:
    st.subheader("Score a transaction")
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Transaction amount (₹)", min_value=1.0, value=2500.0, step=100.0)
        velocity = st.number_input("Transactions in last 1 hour", min_value=0, value=1)
        failed = st.number_input("Failed attempts in last 24h", min_value=0, value=0)
    with c2:
        device = st.slider("Device reputation (0 = poor, 1 = strong)", 0.0, 1.0, 0.85)
        foreign = st.selectbox("Foreign IP", [0,1], format_func=lambda x: "Yes" if x else "No")
        returns = st.number_input("Past returns / disputes", min_value=0, value=1)
    with c3:
        hour = st.slider("Transaction hour", 0, 23, 14)
        account_age = st.number_input("Account age (days)", min_value=1, value=180)

    row = {
        "TransactionAmt": amount,
        "velocity_1h": velocity,
        "failed_attempts": failed,
        "device_reputation": device,
        "is_foreign": foreign,
        "past_returns": returns,
        "hour": hour,
        "is_night": int(hour >= 22 or hour < 5),
        "account_age_days": account_age,
    }
    input_df = pd.DataFrame([row])
    for f in features:
        if f not in input_df.columns:
            input_df[f] = np.nan
    input_df = input_df[features]

    threshold = st.slider("Operational threshold", 0.01, 0.99, default_threshold, 0.01)

    if st.button("Run risk assessment", type="primary"):
        p = float(model.predict_proba(input_df)[:,1][0])
        if p >= threshold:
            decision = "REVIEW / STEP-UP"
            box = st.error
        else:
            decision = "ALLOW"
            box = st.success
        box(f"{decision}  •  estimated fraud probability: {p:.2%}")
        reasons = reason_codes(row)
        st.write("**Reason codes:**", ", ".join(reasons))
        st.caption("This prototype only supports defensive decisions. It does not generate attack instructions, exploit payloads, or evasion strategies.")

with tab2:
    st.subheader("Held-out test-set evidence")
    if saved_metrics:
        cols = st.columns(6)
        cols[0].metric("ROC-AUC", f"{saved_metrics.get('roc_auc', 0):.3f}")
        cols[1].metric("PR-AUC", f"{saved_metrics.get('pr_auc', 0):.3f}")
        cols[2].metric("Precision", f"{saved_metrics.get('precision', 0):.2%}")
        cols[3].metric("Recall", f"{saved_metrics.get('recall', 0):.2%}")
        cols[4].metric("F1", f"{saved_metrics.get('f1', 0):.3f}")
        cols[5].metric("FPR", f"{saved_metrics.get('false_positive_rate', 0):.2%}")
        st.info(f"Operating threshold selected by minimum estimated business cost: {saved_metrics.get('threshold', default_threshold):.3f}")
        st.write("**Dataset note:**", artifact.get("dataset_note", ""))
        st.write("**Estimated false-positive cost:** ₹", saved_metrics.get("false_positives", 0) * saved_metrics.get("fp_cost_inr", 100))
        st.write("**Estimated false-negative cost:** ₹", saved_metrics.get("false_negatives", 0) * saved_metrics.get("fn_cost_inr", 1500))
    pred_file = Path("outputs/test_predictions.csv")
    if pred_file.exists():
        pred = pd.read_csv(pred_file)
        t = st.slider("Review threshold", 0.01, 0.99, default_threshold, 0.01, key="eval_threshold")
        yhat = (pred["p_fraud"] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(pred["y_true"], yhat, labels=[0,1]).ravel()
        a,b,c,d = st.columns(4)
        a.metric("False positives", int(fp))
        b.metric("False negatives", int(fn))
        c.metric("Precision", f"{precision_score(pred['y_true'], yhat, zero_division=0):.2%}")
        d.metric("Recall", f"{recall_score(pred['y_true'], yhat, zero_division=0):.2%}")
        st.dataframe(pd.DataFrame({
            "Actual": pred["y_true"].map({0:"Legitimate",1:"Fraud"}),
            "Risk probability": pred["p_fraud"].round(4),
            "Decision": np.where(yhat, "REVIEW/STEP-UP", "ALLOW")
        }).head(100), use_container_width=True)

with tab3:
    st.subheader("What to show the Razorpay reviewer")
    st.markdown("""
1. Public GitHub repository with clean file names.
2. Architecture diagram in `docs/ARCHITECTURE.md`.
3. Held-out test metrics generated by the training command.
4. False-positive and false-negative cost assumptions.
5. A short demo showing a low-risk and high-risk transaction.
6. A 5-minute pitch covering problem → approach → evidence → business impact → limitations.
7. Explicitly label synthetic/demo data and never claim demo metrics as production performance.
""")
    st.json(saved_metrics)
