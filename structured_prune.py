"""
Structured pruning: shrink Dense(16)->Dense(8) to Dense(8)->Dense(4)
by keeping the highest-L1-norm neurons and transplanting their weights.
Run from project root: python3 structured_prune.py
"""
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

BASE_H5 = os.path.join(MODEL_DIR, "anomaly_model.h5")
STRUCT_PRUNED_H5 = os.path.join(MODEL_DIR, "anomaly_model_structpruned.h5")

TARGET_L1 = 8   # was 16
TARGET_L2 = 4   # was 8


def rank_neurons(W, b, k):
    """Rank neurons of a Dense layer by L1 norm of (weights + bias), return indices of top-k."""
    l1 = np.sum(np.abs(W), axis=0) + np.abs(b)
    top_k_idx = np.argsort(l1)[::-1][:k]
    return np.sort(top_k_idx)  # keep original order for reproducibility


def build_pruned_model():
    base = keras.models.load_model(BASE_H5)
    W1, b1 = base.layers[0].get_weights()   # Dense(16): shape (5,16), (16,)
    W2, b2 = base.layers[1].get_weights()   # Dense(8):  shape (16,8), (8,)
    W3, b3 = base.layers[2].get_weights()   # Dense(5):  shape (8,5), (5,)

    idx1 = rank_neurons(W1, b1, TARGET_L1)
    idx2 = rank_neurons(W2, b2, TARGET_L2)

    print("Kept layer-1 neuron indices:", idx1.tolist())
    print("Kept layer-2 neuron indices:", idx2.tolist())

    # New weight matrices, sliced along the pruned dimensions
    W1_new = W1[:, idx1]                # (5, 8)
    b1_new = b1[idx1]                   # (8,)
    W2_new = W2[idx1][:, idx2]          # (8, 4)  <- slice rows by idx1, cols by idx2
    b2_new = b2[idx2]                   # (4,)
    W3_new = W3[idx2]                   # (4, 5)  <- slice rows by idx2
    b3_new = b3                         # (5,) unchanged

    new_model = keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(TARGET_L1, activation="relu"),
        keras.layers.Dense(TARGET_L2, activation="relu"),
        keras.layers.Dense(5, activation="softmax"),
    ])
    new_model.build(input_shape=(None, 5))
    new_model.layers[0].set_weights([W1_new, b1_new])
    new_model.layers[1].set_weights([W2_new, b2_new])
    new_model.layers[2].set_weights([W3_new, b3_new])

    return new_model


if __name__ == "__main__":
    model = build_pruned_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Quick eval before fine-tuning (weights transplanted, no retraining yet)
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    X = df[["value", "interval_ms", "id", "delta", "repeat_count"]].values.astype("float32")
    y = df["label"].values.astype("int32")
    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(DATA_DIR, "norm_std.npy"))
    X_norm = (X - mean) / std
    _, X_test, _, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Structurally-pruned (no fine-tune) accuracy: {acc:.4f}")

    # Fine-tune to recover accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16,
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Structurally-pruned (after fine-tune) accuracy: {acc:.4f}")

    model.save(STRUCT_PRUNED_H5)
    print("Saved", STRUCT_PRUNED_H5)
