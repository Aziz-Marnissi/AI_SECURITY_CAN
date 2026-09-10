import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

df = pd.read_csv("dataset.csv")
X = df[["value", "interval_ms", "id", "delta", "repeat_count"]].values.astype("float32")
y = df["label"].values.astype("int32")

mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")
X_norm = (X - mean) / std

_, X_test, _, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)

# Reshape to match model input shape (N,1,1,5) expected by X-CUBE-AI
X_test_r = X_test.reshape(-1, 1, 1, 5).astype("float32")

# One-hot encode labels to match softmax output shape (N,1,1,5)
y_onehot = np.eye(5, dtype="float32")[y_test].reshape(-1, 1, 1, 5)

np.save("val_input.npy", X_test_r)
np.save("val_output.npy", y_onehot)
print("Saved val_input.npy", X_test_r.shape, "and val_output.npy", y_onehot.shape)
