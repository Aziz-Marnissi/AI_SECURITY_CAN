"""
Quantize the structurally-pruned model to int8, then compare all three variants:
  1. Baseline (fp32, unpruned, 16-8 architecture)
  2. Unstructured pruned + quantized (int8, 16-8 architecture, 53.8% sparse)
  3. Structurally pruned + quantized (int8, 8-4 architecture)
Run from project root: python3 compare_all.py
"""
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
PLOT_DIR = os.path.join(ROOT, "plots")

STRUCT_H5 = os.path.join(MODEL_DIR, "anomaly_model_structpruned.h5")
STRUCT_TFLITE = os.path.join(MODEL_DIR, "model_structpruned.tflite")

BASELINE_TFLITE = os.path.join(MODEL_DIR, "model_baseline.tflite")
PRUNED_QUANT_TFLITE = os.path.join(MODEL_DIR, "model.tflite")

X_val = np.load(os.path.join(DATA_DIR, "val_input.npy")).astype("float32")


def rep_data_gen():
    for i in range(min(300, len(X_val))):
        yield [X_val[i:i+1]]


def quantize_struct_model():
    model = keras.models.load_model(STRUCT_H5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(STRUCT_TFLITE, "wb") as f:
        f.write(tflite_model)
    print("Saved", STRUCT_TFLITE, f"({len(tflite_model)} bytes)")


def file_size(path):
    return os.path.getsize(path)


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
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def measure_accuracy(tflite_path, X, y, quantized):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if quantized:
        scale, zero_point = input_details["quantization"]
        X_in = (X / scale + zero_point).astype(input_details["dtype"])
    else:
        X_in = X.astype(input_details["dtype"])

    correct = 0
    for i in range(len(X_in)):
        interpreter.set_tensor(input_details["index"], X_in[i:i+1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        pred = np.argmax(out[0])
        if pred == y[i]:
            correct += 1
    return correct / len(X_in)


if __name__ == "__main__":
    quantize_struct_model()

    y_val = np.load(os.path.join(DATA_DIR, "val_output.npy"))

    variants = [
        ("Baseline\n(fp32, 16-8)", BASELINE_TFLITE, False),
        ("Unstructured pruned\n+quantized (int8, 16-8)", PRUNED_QUANT_TFLITE, True),
        ("Structural pruned\n+quantized (int8, 8-4)", STRUCT_TFLITE, True),
    ]

    results = {}
    for name, path, quantized in variants:
        size_bytes = file_size(path)
        ram_bytes, interpreter = estimate_ram(path)
        lat_mean, lat_std = measure_latency(interpreter, X_val, quantized=quantized)
        acc = measure_accuracy(path, X_val, y_val, quantized)
        results[name] = {
            "size_kb": size_bytes / 1024,
            "ram_kb": ram_bytes / 1024,
            "latency_ms": lat_mean,
            "latency_std": lat_std,
            "accuracy": acc,
        }
        print(f"{name.replace(chr(10),' ')}: size={size_bytes/1024:.2f}KB ram={ram_bytes/1024:.2f}KB "
              f"lat={lat_mean:.4f}ms acc={acc:.4f}")

    labels = list(results.keys())
    sizes = [results[k]["size_kb"] for k in labels]
    rams = [results[k]["ram_kb"] for k in labels]
    lats = [results[k]["latency_ms"] for k in labels]
    accs = [results[k]["accuracy"] * 100 for k in labels]
    colors = ["steelblue", "seagreen", "darkorange"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].bar(labels, sizes, color=colors)
    axes[0, 0].set_ylabel("KB")
    axes[0, 0].set_title("Model Size (flash)")

    axes[0, 1].bar(labels, rams, color=colors)
    axes[0, 1].set_ylabel("KB")
    axes[0, 1].set_title("Estimated RAM (tensor arena)")

    axes[1, 0].bar(labels, lats, color=colors)
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].set_title("Inference Latency (host CPU)")

    axes[1, 1].bar(labels, accs, color=colors)
    axes[1, 1].set_ylabel("%")
    axes[1, 1].set_ylim(min(accs) - 2, 100)
    axes[1, 1].set_title("Accuracy")

    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "compare_all_variants.png"), dpi=150)
    plt.close()

    report_path = os.path.join(ROOT, "compare_all_report.txt")
    with open(report_path, "w") as f:
        f.write("SecureAI_IDS: Baseline vs Unstructured vs Structural pruning\n")
        f.write("=" * 60 + "\n\n")
        for name in labels:
            r = results[name]
            f.write(f"{name.replace(chr(10), ' ')}\n")
            f.write(f"  Size (flash):   {r['size_kb']:.2f} KB\n")
            f.write(f"  RAM estimate:   {r['ram_kb']:.2f} KB\n")
            f.write(f"  Latency:        {r['latency_ms']:.4f} +/- {r['latency_std']:.4f} ms\n")
            f.write(f"  Accuracy:       {r['accuracy']*100:.2f}%\n\n")

    print("\nSaved plots/compare_all_variants.png")
    print("Saved", report_path)
