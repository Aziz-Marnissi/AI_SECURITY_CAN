"""
Export C array + val samples from the baseline fp32 model (model_baseline.tflite).
"""
import os
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

TFLITE_PATH = os.path.join(MODEL_DIR, "model_baseline.tflite")
CC_PATH = os.path.join(MODEL_DIR, "model_data_baseline.cc")
H_PATH = os.path.join(MODEL_DIR, "model_data_baseline.h")
VAL_CC = os.path.join(MODEL_DIR, "val_samples_baseline.cc")
VAL_H = os.path.join(MODEL_DIR, "val_samples_baseline.h")

with open(TFLITE_PATH, "rb") as f:
    tflite_model = f.read()

with open(CC_PATH, "w") as f:
    f.write('#include "model_data_baseline.h"\n\n')
    f.write("alignas(8) const unsigned char g_model[] = {\n")
    for i, b in enumerate(tflite_model):
        f.write(f"0x{b:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n")

with open(H_PATH, "w") as f:
    f.write("#ifndef MODEL_DATA_BASELINE_H\n#define MODEL_DATA_BASELINE_H\n\n")
    f.write("extern const unsigned char g_model[];\n")
    f.write("extern const unsigned int g_model_len;\n\n")
    f.write("#endif\n")

print("Saved", CC_PATH, f"({len(tflite_model)} bytes)")

N_SAMPLES = 1596
X_val = np.load(os.path.join(DATA_DIR, "val_input.npy")).astype("float32")
y_val = np.load(os.path.join(DATA_DIR, "val_output.npy")).astype("int32")

rng = np.random.default_rng(42)
idx = rng.choice(len(X_val), size=min(N_SAMPLES, len(X_val)), replace=False)
X_sub = X_val[idx]
y_sub = y_val[idx]

# fp32 model: no quantization, store as floats directly
with open(VAL_CC, "w") as f:
    f.write('#include "val_samples_baseline.h"\n\n')
    f.write(f"const int NUM_VAL_SAMPLES = {len(X_sub)};\n")
    f.write("const float val_samples[][5] = {\n")
    for row in X_sub:
        f.write("  {" + ", ".join(f"{v:.6f}f" for v in row) + "},\n")
    f.write("};\n\n")
    f.write("const uint8_t val_labels[] = {\n  ")
    f.write(", ".join(str(v) for v in y_sub))
    f.write("\n};\n")

with open(VAL_H, "w") as f:
    f.write("#ifndef VAL_SAMPLES_BASELINE_H\n#define VAL_SAMPLES_BASELINE_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("extern const int NUM_VAL_SAMPLES;\n")
    f.write("extern const float val_samples[][5];\n")
    f.write("extern const uint8_t val_labels[];\n\n")
    f.write("#endif\n")

print("Saved", VAL_CC, "and", VAL_H, f"with {len(X_sub)} samples (float32, unquantized)")
