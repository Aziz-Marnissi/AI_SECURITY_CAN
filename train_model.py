import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ["Normal", "Spike", "Flooding", "Replay", "Spoofed_ID"]

df = pd.read_csv("dataset.csv")

X = df[["value", "interval_ms", "id", "delta", "repeat_count"]].values.astype("float32")
y = df["label"].values.astype("int32")

mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X - mean) / std

np.save("norm_mean.npy", mean)
np.save("norm_std.npy", std)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)

model = keras.Sequential([
    keras.layers.Input(shape=(5,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(5, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=80, batch_size=16, validation_data=(X_test, y_test),
                     callbacks=[early_stop], verbose=2)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

model.save("anomaly_model.h5")

# ---- Plot 1: Loss curve ----
plt.figure(figsize=(6,5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("plot_loss.png", dpi=150)
plt.close()

# ---- Plot 2: Accuracy curve ----
plt.figure(figsize=(6,5))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("plot_accuracy.png", dpi=150)
plt.close()

# ---- Plot 3: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(7,6))
disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plot_confusion_matrix.png", dpi=150)
plt.close()

# ---- Plot 4: Per-class ROC/AUC (one-vs-rest) ----
from sklearn.metrics import roc_curve, auc
y_test_bin = label_binarize(y_test, classes=[0,1,2,3,4])
plt.figure(figsize=(7,6))
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_prob[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"--", color="gray")
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("plot_roc.png", dpi=150)
plt.close()

print("Saved: plot_loss.png, plot_accuracy.png, plot_confusion_matrix.png, plot_roc.png")
print("Mean:", mean, "Std:", std)
