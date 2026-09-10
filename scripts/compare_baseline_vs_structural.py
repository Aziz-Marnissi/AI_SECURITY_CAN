"""
Compare Baseline (fp32, 16-8) vs Structurally Pruned + Quantized (int8, 8-4).
Run from project root: python3 compare_baseline_vs_structural.py
"""
import os
import time
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
PLOT_DIR = os.path.join(ROOT, "plots")

BASELINE_TFLITE = os.path.join(MODEL_DIR, "model_baseline.tflite")
STRUCT_TFLITE = os.path.join(MODEL_DIR, "model_structpruned.tflite")

X_val = np.load(os.path.join(DATA_DIR, "val_input.npy")).astype("float32")
y_val = np.load(os.path.join(DATA_DIR, "val_output.npy"))


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
        X_in = (X / scale + zero_point).astype(input_details["dtype"])
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
    variants = [
        ("Baseline\n(fp32, 16-8)", BASELINE_TFLITE, False),
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
    colors = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

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
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "compare_baseline_vs_structural.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    report_path = os.path.join(ROOT, "compare_baseline_vs_structural_report.txt")
    with open(report_path, "w") as f:
        f.write("SecureAI_IDS: Baseline vs Structural Pruned+Quantized\n")
        f.write("=" * 55 + "\n\n")
        for name in labels:
            r = results[name]
            f.write(f"{name.replace(chr(10), ' ')}\n")
            f.write(f"  Size (flash):   {r['size_kb']:.2f} KB\n")
            f.write(f"  RAM estimate:   {r['ram_kb']:.2f} KB\n")
            f.write(f"  Latency:        {r['latency_ms']:.4f} +/- {r['latency_std']:.4f} ms\n")
            f.write(f"  Accuracy:       {r['accuracy']*100:.2f}%\n\n")

    print("\nSaved", out_path)
    print("Saved", report_path)
