"""
SecureAI_IDS pipeline: dataset generation -> training -> validation export -> TFLite conversion.
Run from project root: python3 run_pipeline.py
Creates organized subfolders:
  data/    -> dataset.csv, norm_mean.npy, norm_std.npy, val_input.npy, val_output.npy
  models/  -> anomaly_model.h5, model.tflite, model_data.cc (C array for ESP32)
  plots/   -> plot_loss.png, plot_accuracy.png, plot_confusion_matrix.png, plot_roc.png
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
PLOT_DIR = os.path.join(ROOT, "plots")
for d in (DATA_DIR, MODEL_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

CLASS_NAMES = ["Normal", "Spike", "Flooding", "Replay", "Spoofed_ID"]
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")

def generate_dataset():
    np.random.seed(42)
    N_NORMAL = 6000
    N_ANOMALY_EACH = 500
    rows = []
    prev_val = None
    repeat_count = 0

    def push(val, interval, msg_id, label):
        nonlocal prev_val, repeat_count
        if prev_val is not None and abs(val - prev_val) < 0.05:
            repeat_count += 1
        else:
            repeat_count = 0
        delta = 0.0 if prev_val is None else abs(val - prev_val)
        rows.append([val, interval, msg_id, delta, repeat_count, label])
        prev_val = val

    baselines = [22.0, 25.0, 27.5]
    for b in baselines:
        val = b
        id_cycle = [1, 2, 3]
        n = N_NORMAL // len(baselines)
        for i in range(n):
            val += np.random.uniform(-0.6, 0.6)
            val = np.clip(val, b - 5, b + 5)
            interval = np.random.uniform(85, 115)
            msg_id = id_cycle[i % 3]
            push(val, interval, msg_id, 0)

    for _ in range(N_ANOMALY_EACH):
        val = np.random.choice([np.random.uniform(150, 300), np.random.uniform(-100, -50)])
        interval = np.random.uniform(85, 115)
        msg_id = np.random.choice([1, 2, 3])
        push(val, interval, msg_id, 1)

    for _ in range(N_ANOMALY_EACH):
        val = np.random.uniform(20, 30)
        interval = np.random.uniform(0.5, 10)
        msg_id = np.random.choice([1, 2, 3])
        push(val, interval, msg_id, 2)

    for _ in range(N_ANOMALY_EACH // 30):
        replay_val = np.random.uniform(20, 30)
        for _ in range(30):
            interval = np.random.uniform(85, 115)
            msg_id = np.random.choice([1, 2, 3])
            push(replay_val, interval, msg_id, 3)

    for _ in range(N_ANOMALY_EACH):
        val = np.random.uniform(20, 30)
        interval = np.random.uniform(85, 115)
        msg_id = np.random.choice([9, 42, 77, 99])
        push(val, interval, msg_id, 4)

    df = pd.DataFrame(rows, columns=["value", "interval_ms", "id", "delta", "repeat_count", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATASET_CSV, index=False)
    print(df.label.value_counts())
    print("Saved", DATASET_CSV, "with", len(df), "rows")
    return df

def train():
    df = pd.read_csv(DATASET_CSV)
    X = df[["value", "interval_ms", "id", "delta", "repeat_count"]].values.astype("float32")
    y = df["label"].values.astype("int32")

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    np.save(os.path.join(DATA_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(DATA_DIR, "norm_std.npy"), std)

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

    history = model.fit(X_train, y_train, epochs=80, batch_size=16,
                         validation_data=(X_test, y_test), callbacks=[early_stop], verbose=2)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    model.save(os.path.join(MODEL_DIR, "anomaly_model.h5"))
    model.save(os.path.join(MODEL_DIR, "anomaly_model.keras"))

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_loss.png"), dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_accuracy.png"), dpi=150); plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_confusion_matrix.png"), dpi=150); plt.close()

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC Curves (One-vs-Rest)"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "plot_roc.png"), dpi=150); plt.close()

    print("Saved plots to", PLOT_DIR)
    print("Mean:", mean, "Std:", std)
    return X_norm, y

def prune(X_norm, y):
    model = keras.models.load_model(os.path.join(MODEL_DIR, "anomaly_model.h5"))
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    batch_size = 16
    epochs = 20
    end_step = (len(X_train) // batch_size) * epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.6,
            begin_step=0, end_step=end_step
        )
    }
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_for_pruning.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(X_test, y_test), callbacks=callbacks, verbose=2)

    loss, acc = model_for_pruning.evaluate(X_test, y_test)
    print(f"Pruned test accuracy: {acc:.4f}")

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    pruned_path = os.path.join(MODEL_DIR, "anomaly_model_pruned.h5")
    model_for_export.save(pruned_path)
    print("Saved", pruned_path)
    return model_for_export

def export_validation(X_norm, y):
    _, X_test, _, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)
    np.save(os.path.join(DATA_DIR, "val_input.npy"), X_test.astype("float32"))
    np.save(os.path.join(DATA_DIR, "val_output.npy"), y_test.astype("int32"))
    print("Saved val_input.npy", X_test.shape, "and val_output.npy", y_test.shape)

def convert_tflite(X_norm, model_path=None):
    model_path = model_path or os.path.join(MODEL_DIR, "anomaly_model.h5")
    model = keras.models.load_model(model_path)

    def rep_data_gen():
        for i in range(min(300, len(X_norm))):
            yield [X_norm[i:i+1].astype("float32")]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = os.path.join(MODEL_DIR, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Saved", tflite_path, f"({len(tflite_model)} bytes)")

    cc_path = os.path.join(MODEL_DIR, "model_data.cc")
    with open(cc_path, "w") as f:
        f.write("#include \"model_data.h\"\n\n")
        f.write("alignas(8) const unsigned char g_model[] = {\n")
        for i, b in enumerate(tflite_model):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n")

    h_path = os.path.join(MODEL_DIR, "model_data.h")
    with open(h_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("extern const unsigned char g_model[];\n")
        f.write("extern const unsigned int g_model_len;\n\n")
        f.write("#endif\n")

    print("Saved", cc_path, "and", h_path, "for ESP32/PlatformIO")


if __name__ == "__main__":
    generate_dataset()
    X_norm, y = train()
    export_validation(X_norm, y)
    prune(X_norm, y)
    convert_tflite(X_norm, model_path=os.path.join(MODEL_DIR, "anomaly_model_pruned.h5"))
    print("\nDone. Layout:")
    print(f"  {DATA_DIR}/")
    print(f"  {MODEL_DIR}/  (includes model.tflite, model_data.cc/.h for ESP32)")
    print(f"  {PLOT_DIR}/")
