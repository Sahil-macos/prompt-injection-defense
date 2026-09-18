import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# MODEL 2 — FAST ROBERTA SECONDARY VERIFICATION
# ============================================================

print("=" * 70)
print("MODEL 2 — FAST RoBERTa SECONDARY VERIFICATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.csv"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "validation.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "security",
    "model2",
    "saved_model"
)


# ============================================================
# FAST SETTINGS
# ============================================================

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 4

EVAL_BATCH_SIZE = 8

EPOCHS = 2

LEARNING_RATE = 2e-5


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = "mps"

elif torch.cuda.is_available():

    DEVICE = "cuda"

else:

    DEVICE = "cpu"


print("\nTraining device:", DEVICE)

print("Max sequence length:", MAX_LENGTH)

print("Training batch size:", TRAIN_BATCH_SIZE)

print("Epochs:", EPOCHS)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    df = pd.read_csv(file_path)

    if "text" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'text' column."
        )

    if "label" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'label' column."
        )

    df = df[
        [
            "text",
            "label"
        ]
    ].copy()

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    df = df[
        df["label"].isin([0, 1])
    ].reset_index(drop=True)

    return Dataset.from_pandas(
        df,
        preserve_index=False
    )


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking dataset files...")

for file_path in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}"
        )

    print("Found:", file_path)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = load_data(
    TRAIN_FILE
)

val_dataset = load_data(
    VAL_FILE
)

test_dataset = load_data(
    TEST_FILE
)


print("\nDataset sizes:")

print("Train:", len(train_dataset))

print("Validation:", len(val_dataset))

print("Test:", len(test_dataset))


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTraining label distribution:")

print(
    pd.Series(
        train_dataset["label"]
    )
    .value_counts()
    .sort_index()
)

print("\nValidation label distribution:")

print(
    pd.Series(
        val_dataset["label"]
    )
    .value_counts()
    .sort_index()
)

print("\nTest label distribution:")

print(
    pd.Series(
        test_dataset["label"]
    )
    .value_counts()
    .sort_index()
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading RoBERTa tokenizer...")

tokenizer = (
    RobertaTokenizerFast
    .from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("\nTokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


print("Tokenizing validation data...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


print("Tokenizing test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ============================================================
# LOAD ROBERTA
# ============================================================

print("\nLoading pretrained RoBERTa...")

model = (
    RobertaForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "BENIGN",
            1: "MALICIOUS"
        },
        label2id={
            "BENIGN": 0,
            "MALICIOUS": 1
        }
    )
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits = eval_pred.predictions

    labels = eval_pred.label_ids

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1
    ).numpy()

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    try:

        roc_auc = roc_auc_score(
            labels,
            probabilities[:, 1]
        )

    except ValueError:

        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\nPreparing training configuration...")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_steps=100,

    save_total_limit=1,

    report_to="none",

    fp16=False,

    dataloader_num_workers=0
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("\n")

print("=" * 70)

print("STARTING FAST MODEL 2 TRAINING")

print("=" * 70)

print("\n")

trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\n")

print("=" * 70)

print("VALIDATION RESULTS")

print("=" * 70)


validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)


for key, value in validation_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# FINAL TEST
# ============================================================

print("\n")

print("=" * 70)

print("FINAL TEST RESULTS")

print("=" * 70)


test_results = trainer.evaluate(
    eval_dataset=test_dataset
)


for key, value in test_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

prediction_output = trainer.predict(
    test_dataset
)

logits = prediction_output.predictions

labels = prediction_output.label_ids

probabilities = torch.softmax(
    torch.tensor(logits),
    dim=1
).numpy()

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")

print("=" * 70)

print("CONFUSION MATRIX")

print("=" * 70)

print("\nRows = Actual")

print("Columns = Predicted\n")


cm = confusion_matrix(
    labels,
    predictions
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")

print("=" * 70)

print("CLASSIFICATION REPORT")

print("=" * 70)

print()

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "BENIGN",
            "MALICIOUS"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = pd.DataFrame({

    "actual_label": labels,

    "predicted_label": predictions,

    "benign_probability":
        probabilities[:, 0],

    "malicious_probability":
        probabilities[:, 1]
})


prediction_file = os.path.join(
    BASE_DIR,
    "security",
    "model2",
    "test_predictions.csv"
)


results_df.to_csv(
    prediction_file,
    index=False
)


print(
    "\nTest predictions saved to:"
)

print(
    prediction_file
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n")

print("=" * 70)

print("SAVING MODEL 2")

print("=" * 70)


trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print("\nModel 2 saved to:")

print(OUTPUT_DIR)


# ============================================================
# DONE
# ============================================================

print("\n")

print("=" * 70)

print("MODEL 2 TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)