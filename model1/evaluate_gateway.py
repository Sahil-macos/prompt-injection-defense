from pathlib import Path
import sys
import time
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/Users/sahilbaliyan/Documents/prompt injection"
)

MODEL1_DIR = BASE_DIR / "model1"

TEST_CSV = Path(
    "/Users/sahilbaliyan/Downloads/"
    "NeurAlchemy-Prompt-Injection-full/"
    "processed/test.csv"
)

OUTPUT_CSV = (
    BASE_DIR / "gateway_evaluation_results.csv"
)

ERROR_CSV = (
    BASE_DIR / "gateway_evaluation_errors.csv"
)


# ============================================================
# IMPORT SECURITY GATEWAY
# ============================================================

sys.path.insert(
    0,
    str(MODEL1_DIR)
)

import security_gateway as gateway


# ============================================================
# EVALUATION MODE
#
# IMPORTANT:
#
# We DO NOT want Llama generating responses for every
# test sample.
#
# The purpose of this script is to evaluate the security
# gateway itself.
#
# Llama is therefore replaced with a dummy function.
# ============================================================

def evaluation_llama_response(prompt):
    return ""


# Replace the actual Llama function.
#
# Your traceback showed that security_gateway.py calls:
#
#     generate_llama_response(prompt)
#
# Therefore we patch THAT function.
# ============================================================

if hasattr(
    gateway,
    "generate_llama_response"
):

    gateway.generate_llama_response = (
        evaluation_llama_response
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "generate_llama_response() was not found "
        "inside security_gateway.py."
    )

    print(
        "Llama may still be called during evaluation."
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"\nTest CSV not found:\n{TEST_CSV}\n"
    )


df = pd.read_csv(
    TEST_CSV
)


print("\n")
print("=" * 70)
print("SECURITY GATEWAY EVALUATION")
print("=" * 70)

print(
    f"Test file: {TEST_CSV}"
)

print(
    f"Total samples: {len(df)}"
)

print(
    f"Columns: {list(df.columns)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "text" not in df.columns:

    raise ValueError(
        "Expected a 'text' column in test.csv."
    )


if "label" not in df.columns:

    raise ValueError(
        "Expected a 'label' column in test.csv."
    )


# ============================================================
# LABEL CONVERSION
#
# 0 = BENIGN
# 1 = MALICIOUS
# ============================================================

def normalize_label(value):

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {
            "benign",
            "safe",
            "0"
        }:

            return 0

        if value in {
            "malicious",
            "unsafe",
            "1"
        }:

            return 1

    try:

        value = int(value)

        if value in {0, 1}:

            return value

    except Exception:

        pass

    raise ValueError(
        f"Unknown label: {value}"
    )


# ============================================================
# TRUE LABELS
# ============================================================

y_true = []

valid_rows = []

label_errors = []


for index, row in df.iterrows():

    try:

        label = normalize_label(
            row["label"]
        )

        y_true.append(label)

        valid_rows.append(
            index
        )

    except Exception as e:

        label_errors.append(
            {
                "index": index,
                "error": str(e)
            }
        )


# ============================================================
# RUN SECURITY GATEWAY
# ============================================================

y_pred = []

prediction_indices = []

allow_count = 0

block_count = 0

model2_count = 0

total_time = 0.0

errors = []


total_samples = len(valid_rows)


print("\n")
print(
    "Starting gateway evaluation..."
)

print(
    "Llama generation: DISABLED"
)

print(
    "Model 2 RoBERTa: ENABLED"
)

print(
    "\n"
)


# ============================================================
# EVALUATION LOOP
# ============================================================

for position, index in enumerate(
    valid_rows,
    start=1
):

    row = df.loc[index]

    prompt = str(
        row["text"]
    )

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # SECURITY GATEWAY
        # ----------------------------------------------------

        result = gateway.security_check(
            prompt
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed


        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        final_decision = str(
            result.get(
                "final_decision",
                ""
            )
        ).upper().strip()


        # ----------------------------------------------------
        # CONVERT DECISION TO CLASS
        #
        # ALLOW = BENIGN
        # BLOCK = MALICIOUS
        # ----------------------------------------------------

        if final_decision == "ALLOW":

            prediction = 0

            allow_count += 1

        elif final_decision == "BLOCK":

            prediction = 1

            block_count += 1

        else:

            raise ValueError(
                f"Unknown final decision: "
                f"{final_decision}"
            )


        # ----------------------------------------------------
        # MODEL 2
        # ----------------------------------------------------

        if result.get(
            "model2_used",
            False
        ):

            model2_count += 1


        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        y_pred.append(
            prediction
        )

        prediction_indices.append(
            index
        )


    except KeyboardInterrupt:

        print(
            "\n\nEvaluation stopped by user."
        )

        break


    except Exception as e:

        errors.append(
            {
                "index": index,
                "text": prompt,
                "error": str(e)
            }
        )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        position % 10 == 0
        or position == total_samples
    ):

        elapsed_total = (
            time.perf_counter()
            - (
                start_time
                - total_time
                if total_time
                else start_time
            )
        )

        print(
            f"Processed "
            f"{position}/{total_samples} "
            f"| Errors: {len(errors)} "
            f"| Model 2: {model2_count}",
            flush=True
        )


# ============================================================
# CHECK WHETHER EVALUATION FINISHED
# ============================================================

evaluated_samples = len(
    y_pred
)


if evaluated_samples == 0:

    raise RuntimeError(
        "\nNo samples were successfully evaluated."
    )


# ============================================================
# MATCH TRUE LABELS TO SUCCESSFUL PREDICTIONS
# ============================================================

successful_y_true = []

for index in prediction_indices:

    position = valid_rows.index(
        index
    )

    successful_y_true.append(
        y_true[position]
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    successful_y_true,
    y_pred
)

precision = precision_score(
    successful_y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    successful_y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    successful_y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    successful_y_true,
    y_pred,
    labels=[0, 1]
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# TIMING
# ============================================================

avg_ms = (
    total_time
    / evaluated_samples
    * 1000
)


# ============================================================
# MODEL 2 RATE
# ============================================================

model2_rate = (
    model2_count
    / evaluated_samples
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print(
    f"Total test samples:       {len(df)}"
)

print(
    f"Successfully evaluated:   {evaluated_samples}"
)

print(
    f"Gateway errors:            {len(errors)}"
)

print(
    f"Accuracy:                  {accuracy:.4f}"
)

print(
    f"Precision:                 {precision:.4f}"
)

print(
    f"Recall:                    {recall:.4f}"
)

print(
    f"F1-score:                  {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("CONFUSION MATRIX")
print("-" * 70)

print(
    "                  Predicted"
)

print(
    "                Benign  Malicious"
)

print(
    f"Actual Benign     "
    f"{tn:5d}    "
    f"{fp:5d}"
)

print(
    f"Actual Malicious  "
    f"{fn:5d}    "
    f"{tp:5d}"
)


# ============================================================
# GATEWAY ROUTING
# ============================================================

print("\n")
print("GATEWAY ROUTING")
print("-" * 70)

print(
    f"ALLOW decisions:          {allow_count}"
)

print(
    f"BLOCK decisions:          {block_count}"
)

print(
    f"Model 2 triggered:        {model2_count}"
)

print(
    f"Model 2 trigger rate:     "
    f"{model2_rate:.2f}%"
)

print(
    f"Average gateway time:     "
    f"{avg_ms:.2f} ms"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        successful_y_true,
        y_pred,
        target_names=[
            "BENIGN",
            "MALICIOUS"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE SUMMARY RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "metric": [

            "total_samples",

            "successfully_evaluated",

            "errors",

            "accuracy",

            "precision",

            "recall",

            "f1",

            "true_negatives",

            "false_positives",

            "false_negatives",

            "true_positives",

            "allow_decisions",

            "block_decisions",

            "model2_triggered",

            "model2_trigger_rate_percent",

            "average_gateway_time_ms"
        ],

        "value": [

            len(df),

            evaluated_samples,

            len(errors),

            accuracy,

            precision,

            recall,

            f1,

            tn,

            fp,

            fn,

            tp,

            allow_count,

            block_count,

            model2_count,

            model2_rate,

            avg_ms
        ]
    }
)


results.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE ERRORS
# ============================================================

if errors:

    error_df = pd.DataFrame(
        errors
    )

    error_df.to_csv(
        ERROR_CSV,
        index=False
    )

    print(
        f"\nErrors saved to:\n"
        f"{ERROR_CSV}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)

print(
    "EVALUATION COMPLETE"
)

print("=" * 70)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_CSV}"
)

print(
    "\nLlama was NOT used for generating responses."
)

print(
    "The evaluation measured the Security Gateway."
)

print("=" * 70)