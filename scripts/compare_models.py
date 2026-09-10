"""
Compare baseline (float32, unpruned) vs pruned+quantized (int8) model.
Run from project root: python3 compare_models.py
Produces:
  models/model_baseline.tflite   (float32, unpruned, for fair comparison)
  plots/compare_size.png
  plots/compare_ram.png
  plots/compare_latency.png
  compare_report.txt
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
PLOT_DIR = os.path.join(ROOT, "plots")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_H5 = os.path.join(MODEL_DIR, "anomaly_model.h5")
PRUNED_H5 = os.path.join(MODEL_DIR, "anomaly_model_pruned.h5")
BASELINE_TFLITE = os.path.join(MODEL_DIR, "model_baseline.tflite")
PRUNED_QUANT_TFLITE = os.path.join(MODEL_DIR, "model.tflite")

X_val = np.load(os.path.join(DATA_DIR, "val_input.npy")).astype("float32")

# ---------------- 1. Build float32 baseline .tflite (unpruned, no quantization) ----------------
def export_baseline_tflite():
    model = keras.models.load_model(BASE_H5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(BASELINE_TFLITE, "wb") as f:
        f.write(tflite_model)
    print("Saved", BASELINE_TFLITE, f"({len(tflite_model)} bytes)")

# ---------------- 2. Measure file size (flash usage proxy) ----------------
def file_size(path):
    return os.path.getsize(path)

# ---------------- 3. Estimate RAM (sum of tensor buffer sizes = arena proxy) ----------------
def estimate_ram(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    total_bytes = 0
    for detail in interpreter.get_tensor_details():
        shape = detail["shape"]
        dtype = detail["dtype"]
        itemsize = np.dtype(dtype).itemsize
        total_bytes += int(np.prod(shape)) * itemsize if shape.size > 0 else itemsize
    return total_bytes, interpreter

# ---------------- 4. Measure inference latency ----------------
def measure_latency(interpreter, X, n_runs=200, quantized=False):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if quantized:
        scale, zero_point = input_details["quantization"]
        X_in = (X / scale + zero_point).astype(input_details["dtype"]) if scale != 0 else X.astype(input_details["dtype"])
    else:
        X_in = X.astype(input_details["dtype"])

    n = min(n_runs, len(X_in))
    times = []
    for i in range(n):
        sample = X_in[i:i+1]
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.mean(times), np.std(times)


def count_nonzero_params(h5_path):
    model = keras.models.load_model(h5_path)
    total = 0
    nonzero = 0
    for w in model.get_weights():
        total += w.size
        nonzero += np.count_nonzero(w)
    return total, nonzero


if __name__ == "__main__":
    export_baseline_tflite()

    results = {}
    for name, path, quantized in [
        ("Baseline (fp32, unpruned)", BASELINE_TFLITE, False),
        ("Pruned + Quantized (int8)", PRUNED_QUANT_TFLITE, True),
    ]:
        size_bytes = file_size(path)
        ram_bytes, interpreter = estimate_ram(path)
        lat_mean, lat_std = measure_latency(interpreter, X_val, quantized=quantized)
        results[name] = {
            "size_kb": size_bytes / 1024,
            "ram_kb": ram_bytes / 1024,
            "latency_ms": lat_mean,
            "latency_std": lat_std,
        }
        print(f"{name}: size={size_bytes/1024:.2f} KB, ram~={ram_bytes/1024:.2f} KB, "
              f"latency={lat_mean:.4f}±{lat_std:.4f} ms")

    total_w, nonzero_w = count_nonzero_params(PRUNED_H5)
    sparsity = 100 * (1 - nonzero_w / total_w)
    print(f"\nPruned model weight sparsity: {sparsity:.1f}% "
          f"({nonzero_w}/{total_w} nonzero params)")

    # ---------------- Plots ----------------
    labels = list(results.keys())
    sizes = [results[k]["size_kb"] for k in labels]
    rams = [results[k]["ram_kb"] for k in labels]
    lats = [results[k]["latency_ms"] for k in labels]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, sizes, color=["steelblue", "seagreen"])
    plt.ylabel("Flash / model size (KB)")
    plt.title("Model Size: Baseline vs Pruned+Quantized")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_size.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(labels, rams, color=["steelblue", "seagreen"])
    plt.ylabel("Estimated RAM / tensor arena (KB)")
    plt.title("RAM Usage: Baseline vs Pruned+Quantized")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_ram.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(labels, lats, color=["steelblue", "seagreen"])
    plt.ylabel("Avg inference latency (ms, host CPU)")
    plt.title("Latency: Baseline vs Pruned+Quantized")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_latency.png"), dpi=150)
    plt.close()

    # ---------------- Report ----------------
    report_path = os.path.join(ROOT, "compare_report.txt")
    with open(report_path, "w") as f:
        f.write("SecureAI_IDS: Baseline vs Pruned+Quantized comparison\n")
        f.write("=" * 55 + "\n\n")
        for name in labels:
            r = results[name]
            f.write(f"{name}\n")
            f.write(f"  Size (flash):  {r['size_kb']:.2f} KB\n")
            f.write(f"  RAM estimate:  {r['ram_kb']:.2f} KB\n")
            f.write(f"  Latency:       {r['latency_ms']:.4f} ± {r['latency_std']:.4f} ms (host CPU)\n\n")
        f.write(f"Pruned weight sparsity: {sparsity:.1f}% ({nonzero_w}/{total_w} nonzero)\n")
        f.write("\nNote: latency measured on host CPU (proxy only). "
                "Actual ESP32 inference time/RAM must be measured on-device "
                "via esp-tflite-micro (use tflite_micro benchmark or log micros() around interpreter.Invoke()).\n")

    print("\nSaved plots to", PLOT_DIR)
    print("Saved report to", report_path)
