import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "processed")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "model1",
    "saved_model"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

WORD_MAX_LEN = 128
CHAR_MAX_LEN = 256

WORD_VOCAB_SIZE = 20000
CHAR_VOCAB_SIZE = 100

WORD_EMBED_DIM = 64
CHAR_EMBED_DIM = 32

WORD_CHANNELS = 64
CHAR_CHANNELS = 64

HIDDEN_SIZE = 128

BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# ============================================================
# DATA LOADING
# ============================================================

def load_csv(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(
            f"'text' column missing from {path}"
        )

    if "label" not in df.columns:
        raise ValueError(
            f"'label' column missing from {path}"
        )

    df = df[["text", "label"]].copy()

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df["label"] = (
        pd.to_numeric(
            df["label"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df = df[
        df["label"].isin([0, 1])
    ].reset_index(drop=True)

    return df


print("=" * 70)
print("MODEL 1 — CUSTOM TEXT MVNET")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading datasets...")

train_df = load_csv(TRAIN_FILE)
val_df = load_csv(VAL_FILE)
test_df = load_csv(TEST_FILE)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["label"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["label"].value_counts().sort_index())

# ============================================================
# WORD VOCABULARY
# ============================================================

PAD_WORD = 0
UNK_WORD = 1

def word_tokenize(text):
    return text.lower().split()


def build_word_vocab(texts):

    frequencies = {}

    for text in texts:

        for token in word_tokenize(text):

            frequencies[token] = (
                frequencies.get(token, 0) + 1
            )

    sorted_words = sorted(
        frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )

    vocab = {
        "<PAD>": PAD_WORD,
        "<UNK>": UNK_WORD
    }

    for word, _ in sorted_words[
        :WORD_VOCAB_SIZE - 2
    ]:

        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


print("\nBuilding word vocabulary...")

word_vocab = build_word_vocab(
    train_df["text"].tolist()
)

print(
    "Word vocabulary size:",
    len(word_vocab)
)

# ============================================================
# CHARACTER VOCABULARY
# ============================================================

PAD_CHAR = 0
UNK_CHAR = 1

characters = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-_/\\@#$%^&*()+={}[]<>|`~\t\n"
)

char_vocab = {
    "<PAD>": PAD_CHAR,
    "<UNK>": UNK_CHAR
}

for char in characters:

    if char not in char_vocab:
        char_vocab[char] = len(char_vocab)

print(
    "Character vocabulary size:",
    len(char_vocab)
)

# ============================================================
# ENCODING
# ============================================================

def encode_words(text):

    tokens = word_tokenize(text)

    ids = [
        word_vocab.get(
            token,
            UNK_WORD
        )
        for token in tokens
    ]

    ids = ids[:WORD_MAX_LEN]

    if len(ids) < WORD_MAX_LEN:

        ids += [
            PAD_WORD
        ] * (
            WORD_MAX_LEN - len(ids)
        )

    return ids


def encode_chars(text):

    chars = list(text[:CHAR_MAX_LEN])

    ids = [
        char_vocab.get(
            char,
            UNK_CHAR
        )
        for char in chars
    ]

    if len(ids) < CHAR_MAX_LEN:

        ids += [
            PAD_CHAR
        ] * (
            CHAR_MAX_LEN - len(ids)
        )

    return ids

# ============================================================
# DATASET
# ============================================================

class PromptDataset(Dataset):

    def __init__(self, dataframe):

        self.texts = dataframe[
            "text"
        ].tolist()

        self.labels = dataframe[
            "label"
        ].tolist()

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        text = self.texts[index]

        word_ids = encode_words(text)

        char_ids = encode_chars(text)

        label = self.labels[index]

        return (
            torch.tensor(
                word_ids,
                dtype=torch.long
            ),
            torch.tensor(
                char_ids,
                dtype=torch.long
            ),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )

# ============================================================
# DATALOADERS
# ============================================================

train_dataset = PromptDataset(train_df)
val_dataset = PromptDataset(val_df)
test_dataset = PromptDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# MVNET MODEL
# ============================================================

class MVNet(nn.Module):

    def __init__(
        self,
        word_vocab_size,
        char_vocab_size
    ):

        super().__init__()

        # ----------------------------------------------------
        # WORD VIEW
        # ----------------------------------------------------

        self.word_embedding = nn.Embedding(
            word_vocab_size,
            WORD_EMBED_DIM,
            padding_idx=PAD_WORD
        )

        self.word_conv = nn.Conv1d(
            in_channels=WORD_EMBED_DIM,
            out_channels=WORD_CHANNELS,
            kernel_size=5,
            padding=2
        )

        # ----------------------------------------------------
        # CHARACTER VIEW
        # ----------------------------------------------------

        self.char_embedding = nn.Embedding(
            char_vocab_size,
            CHAR_EMBED_DIM,
            padding_idx=PAD_CHAR
        )

        self.char_conv = nn.Conv1d(
            in_channels=CHAR_EMBED_DIM,
            out_channels=CHAR_CHANNELS,
            kernel_size=5,
            padding=2
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                WORD_CHANNELS + CHAR_CHANNELS,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Dropout(0.30)
        )

        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            2
        )

    def forward(
        self,
        word_ids,
        char_ids
    ):

        # ====================================================
        # WORD VIEW
        # ====================================================

        word_x = self.word_embedding(
            word_ids
        )

        word_x = word_x.transpose(
            1,
            2
        )

        word_x = self.word_conv(
            word_x
        )

        word_x = torch.relu(
            word_x
        )

        word_x = torch.max(
            word_x,
            dim=2
        ).values

        # ====================================================
        # CHARACTER VIEW
        # ====================================================

        char_x = self.char_embedding(
            char_ids
        )

        char_x = char_x.transpose(
            1,
            2
        )

        char_x = self.char_conv(
            char_x
        )

        char_x = torch.relu(
            char_x
        )

        char_x = torch.max(
            char_x,
            dim=2
        ).values

        # ====================================================
        # MULTI-VIEW FUSION
        # ====================================================

        combined = torch.cat(
            [
                word_x,
                char_x
            ],
            dim=1
        )

        fused = self.fusion(
            combined
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        logits = self.classifier(
            fused
        )

        return logits

# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating MVNet...")

model = MVNet(
    len(word_vocab),
    len(char_vocab)
).to(DEVICE)

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

# ============================================================
# LOSS
# ============================================================

class_counts = np.bincount(
    train_df["label"].values,
    minlength=2
)

class_weights = (
    len(train_df)
    /
    (
        2 *
        np.maximum(
            class_counts,
            1
        )
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    for (
        word_ids,
        char_ids,
        labels
    ) in loader:

        word_ids = word_ids.to(
            DEVICE
        )

        char_ids = char_ids.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        logits = model(
            word_ids,
            char_ids
        )

        loss = criterion(
            logits,
            labels
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .tolist()
        )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        roc_auc = 0.0

    return {
        "loss": total_loss /
        max(total_samples, 1),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "labels": all_labels,

        "predictions": all_predictions,

        "probabilities": all_probabilities
    }

# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING MVNET TRAINING")
print("=" * 70)

best_f1 = -1.0

best_path = os.path.join(
    OUTPUT_DIR,
    "mvnet_best.pt"
)

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for (
        word_ids,
        char_ids,
        labels
    ) in train_loader:

        word_ids = word_ids.to(
            DEVICE
        )

        char_ids = char_ids.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            word_ids,
            char_ids
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size

    train_loss = (
        running_loss /
        max(total_samples, 1)
    )

    val_metrics = evaluate(
        model,
        val_loader
    )

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_metrics['loss']:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Precision: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Val Recall: "
        f"{val_metrics['recall']:.4f}"
    )

    print(
        f"Val F1: "
        f"{val_metrics['f1']:.4f}"
    )

    print(
        f"Val ROC-AUC: "
        f"{val_metrics['roc_auc']:.4f}"
    )

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "word_vocab":
                    word_vocab,

                "char_vocab":
                    char_vocab,

                "config": {
                    "word_max_len":
                        WORD_MAX_LEN,

                    "char_max_len":
                        CHAR_MAX_LEN,

                    "word_embed_dim":
                        WORD_EMBED_DIM,

                    "char_embed_dim":
                        CHAR_EMBED_DIM,

                    "word_channels":
                        WORD_CHANNELS,

                    "char_channels":
                        CHAR_CHANNELS,

                    "hidden_size":
                        HIDDEN_SIZE
                },

                "id2label": {
                    0: "BENIGN",
                    1: "MALICIOUS"
                },

                "label2id": {
                    "BENIGN": 0,
                    "MALICIOUS": 1
                }
            },
            best_path
        )

        print(
            "Best MVNet saved:",
            best_path
        )

# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MVNET TEST RESULTS")
print("=" * 70)

checkpoint = torch.load(
    best_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_metrics = evaluate(
    model,
    test_loader
)

print(
    f"\nTest Loss: "
    f"{test_metrics['loss']:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test Recall: "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Test F1: "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Test ROC-AUC: "
    f"{test_metrics['roc_auc']:.4f}"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    test_metrics["labels"],
    test_metrics["predictions"]
)

print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted\n"
)

print(cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
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

prediction_file = os.path.join(
    BASE_DIR,
    "model1",
    "test_predictions.csv"
)

prediction_df = pd.DataFrame({

    "actual_label":
        test_metrics["labels"],

    "predicted_label":
        test_metrics["predictions"],

    "malicious_probability":
        test_metrics["probabilities"]
})

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    "\nTest predictions saved to:"
)

print(prediction_file)

# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)

metrics_to_save = {
    "test_loss":
        test_metrics["loss"],

    "accuracy":
        test_metrics["accuracy"],

    "precision":
        test_metrics["precision"],

    "recall":
        test_metrics["recall"],

    "f1":
        test_metrics["f1"],

    "roc_auc":
        test_metrics["roc_auc"]
}

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_to_save,
        f,
        indent=2
    )

print(
    "Metrics saved to:"
)

print(metrics_file)

# ============================================================
# SAVE VOCABULARIES
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "word_vocab.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        word_vocab,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    os.path.join(
        OUTPUT_DIR,
        "char_vocab.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        char_vocab,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n")
print("=" * 70)
print("MODEL 1 MVNET TRAINING COMPLETED")
print("=" * 70)

print(
    "\nBest model:"
)

print(best_path)

print(
    "\nArchitecture:"
)

print(
    "Word View + Character View "
    "-> Feature Fusion -> Linear Classifier"
)

print(
    "\nModel 1 output:"
)

print(
    "0 = BENIGN"
)

print(
    "1 = MALICIOUS"
)

print(
    "\nNext stage:"
)

print(
    "MVNet score -> Policy Engine "
    "-> Risk Fusion -> RoBERTa Model 2"
)