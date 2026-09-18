import sys
import json
from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/Users/sahilbaliyan/Documents/prompt injection"
)

MODEL1_DIR = BASE_DIR / "model1"

MODEL2_DIR = (
    BASE_DIR /
    "model2" /
    "saved_model"
)

TEST_FILE = (
    BASE_DIR /
    "processed" /
    "test.csv"
)


# ============================================================
# IMPORT MODEL 1
# ============================================================

sys.path.insert(
    0,
    str(MODEL1_DIR)
)

from infer_model1 import predict as mvnet_predict
from policy_engine import check_policy
from risk_fusion import calculate_risk


# ============================================================
# CONFIGURATION
# ============================================================

ROUTING_THRESHOLD = 0.30

ROBERTA_THRESHOLD = 0.50

MAX_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "COMPLETE TWO-STAGE SYSTEM EVALUATION"
)

print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...")

if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_FILE}"
    )

if not MODEL2_DIR.exists():

    raise FileNotFoundError(
        f"Model 2 directory not found:\n{MODEL2_DIR}"
    )

print("Test dataset found.")

print("Model 2 directory found.")


# ============================================================
# LOAD ROBERTA MODEL 2
# ============================================================

print("\nLoading RoBERTa Model 2...")

tokenizer = (
    RobertaTokenizerFast
    .from_pretrained(
        str(MODEL2_DIR)
    )
)

roberta = (
    RobertaForSequenceClassification
    .from_pretrained(
        str(MODEL2_DIR)
    )
)

roberta.to(DEVICE)

roberta.eval()

print(
    "RoBERTa loaded successfully."
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# ROBERTA PREDICTION
# ============================================================

@torch.no_grad()
def roberta_predict(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = roberta(
        **inputs
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    benign_probability = (
        probabilities[0].item()
    )

    malicious_probability = (
        probabilities[1].item()
    )

    if (
        malicious_probability
        >= ROBERTA_THRESHOLD
    ):

        prediction = 1

    else:

        prediction = 0

    return {
        "prediction": prediction,
        "benign_probability":
            benign_probability,
        "malicious_probability":
            malicious_probability
    }


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(
    TEST_FILE
)

texts = (
    df["text"]
    .astype(str)
    .tolist()
)

labels = (
    df["label"]
    .astype(int)
    .tolist()
)

print(
    "Test samples:",
    len(df)
)

print(
    "BENIGN:",
    sum(
        x == 0
        for x in labels
    )
)

print(
    "MALICIOUS:",
    sum(
        x == 1
        for x in labels
    )
)


# ============================================================
# RESULT STORAGE
# ============================================================

final_predictions = []

final_scores = []

mvnet_scores = []

mvnet_predictions = []

policy_scores = []

risk_scores = []

routing_decisions = []

model2_used = []

model2_predictions = []

model2_scores = []


# ============================================================
# COMPLETE PIPELINE
# ============================================================

print("\n")

print("=" * 70)

print(
    "RUNNING COMPLETE PIPELINE"
)

print("=" * 70)


for i, text in enumerate(texts):

    # ========================================================
    # MODEL 1 — MVNET
    # ========================================================

    mvnet_result = mvnet_predict(
        text
    )

    # IMPORTANT:
    # infer_model1.py returns:
    #
    # malicious_probability
    #
    # NOT malicious_score

    mvnet_score = float(
        mvnet_result[
            "malicious_probability"
        ]
    )

    mvnet_prediction = int(
        mvnet_result[
            "prediction"
        ]
    )


    # ========================================================
    # POLICY ENGINE
    # ========================================================

    policy_result = check_policy(
        text
    )

    policy_score = float(
        policy_result[
            "policy_score"
        ]
    )

    matched_rules = (
        policy_result[
            "matched_rules"
        ]
    )


    # ========================================================
    # RISK FUSION
    # ========================================================

    fusion_result = (
        calculate_risk(
            mvnet_score=mvnet_score,

            policy_score=policy_score,

            matched_rules=matched_rules
        )
    )

    risk_score = float(
        fusion_result[
            "risk_score"
        ]
    )


    # ========================================================
    # ROUTING DECISION
    # ========================================================

    if risk_score > ROUTING_THRESHOLD:

        # ====================================================
        # SECONDARY VERIFICATION
        # MODEL 2 — ROBERTA
        # ====================================================

        roberta_result = (
            roberta_predict(
                text
            )
        )

        final_prediction = (
            roberta_result[
                "prediction"
            ]
        )

        final_score = (
            roberta_result[
                "malicious_probability"
            ]
        )

        used_model2 = True

        model2_prediction = (
            roberta_result[
                "prediction"
            ]
        )

        model2_score = (
            roberta_result[
                "malicious_probability"
            ]
        )

        routing = "MODEL_2"

    else:

        # ====================================================
        # LOW RISK
        # MODEL 2 NOT REQUIRED
        # ====================================================

        final_prediction = 0

        final_score = risk_score

        used_model2 = False

        model2_prediction = -1

        model2_score = 0.0

        routing = "ALLOW"


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    final_predictions.append(
        final_prediction
    )

    final_scores.append(
        final_score
    )

    mvnet_scores.append(
        mvnet_score
    )

    mvnet_predictions.append(
        mvnet_prediction
    )

    policy_scores.append(
        policy_score
    )

    risk_scores.append(
        risk_score
    )

    routing_decisions.append(
        routing
    )

    model2_used.append(
        used_model2
    )

    model2_predictions.append(
        model2_prediction
    )

    model2_scores.append(
        model2_score
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        (i + 1) % 100 == 0
    ):

        print(
            f"Processed "
            f"{i + 1}/"
            f"{len(texts)}"
        )


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    final_predictions
)

precision = precision_score(
    labels,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    final_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    labels,
    final_scores
)

cm = confusion_matrix(
    labels,
    final_predictions
)


# ============================================================
# CONFUSION MATRIX VALUES
# ============================================================

tn, fp, fn, tp = (
    cm.ravel()
)


# ============================================================
# ROUTING STATISTICS
# ============================================================

total = len(
    labels
)

model1_only = sum(
    not x
    for x in model2_used
)

sent_to_model2 = sum(
    model2_used
)

routing_percentage = (
    sent_to_model2 /
    total
) * 100


# ============================================================
# FINAL SYSTEM RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "FINAL TWO-STAGE SYSTEM RESULTS"
)

print("=" * 70)

print()

print(
    f"Test Samples:       {total}"
)

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall:             {recall:.4f}"
)

print(
    f"F1 Score:           {f1:.4f}"
)

print(
    f"ROC-AUC:            {roc_auc:.4f}"
)


# ============================================================
# ROUTING STATISTICS
# ============================================================

print("\n")

print("=" * 70)

print(
    "ROUTING STATISTICS"
)

print("=" * 70)

print()

print(
    f"Handled by Model 1 only: "
    f"{model1_only}"
)

print(
    f"Sent to RoBERTa Model 2: "
    f"{sent_to_model2}"
)

print(
    f"Model 2 verification rate: "
    f"{routing_percentage:.2f}%"
)

print()

print(
    f"Routing threshold: "
    f">{ROUTING_THRESHOLD:.2f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")

print("=" * 70)

print(
    "CONFUSION MATRIX"
)

print("=" * 70)

print()

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(cm)

print()

print(
    f"True Negatives:  {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives:  {tp}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")

print("=" * 70)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 70)

print()

print(
    classification_report(
        labels,
        final_predictions,

        target_names=[
            "BENIGN",
            "MALICIOUS"
        ],

        zero_division=0
    )
)


# ============================================================
# SAVE DETAILED PREDICTIONS
# ============================================================

results_df = df.copy()

results_df[
    "mvnet_prediction"
] = mvnet_predictions

results_df[
    "mvnet_malicious_probability"
] = mvnet_scores

results_df[
    "policy_score"
] = policy_scores

results_df[
    "risk_score"
] = risk_scores

results_df[
    "routing"
] = routing_decisions

results_df[
    "model2_used"
] = model2_used

results_df[
    "model2_prediction"
] = model2_predictions

results_df[
    "model2_malicious_probability"
] = model2_scores

results_df[
    "final_prediction"
] = final_predictions

results_df[
    "final_score"
] = final_scores


# ============================================================
# SAVE CSV
# ============================================================

output_file = (
    MODEL1_DIR /
    "full_system_predictions.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {

    "test_samples":
        total,

    "accuracy":
        accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1":
        f1,

    "roc_auc":
        roc_auc,

    "true_negatives":
        int(tn),

    "false_positives":
        int(fp),

    "false_negatives":
        int(fn),

    "true_positives":
        int(tp),

    "model1_only":
        model1_only,

    "model2_used":
        sent_to_model2,

    "model2_verification_rate":
        routing_percentage,

    "routing_threshold":
        ROUTING_THRESHOLD,

    "roberta_threshold":
        ROBERTA_THRESHOLD
}


metrics_file = (
    MODEL1_DIR /
    "full_system_metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n")

print("=" * 70)

print(
    "COMPLETE SYSTEM EVALUATION COMPLETED"
)

print("=" * 70)

print()

print(
    "Predictions saved to:"
)

print(
    output_file
)

print()

print(
    "Metrics saved to:"
)

print(
    metrics_file
)

print()

print(
    "Pipeline:"
)

print(
    "MVNet"
    " -> Policy Engine"
    " -> Risk Fusion"
    " -> 30% Routing"
    " -> RoBERTa Model 2"
)

print()