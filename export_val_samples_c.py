"""
Export N validation samples (int8-quantized) plus true labels as a C array.
"""
import os
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

TFLITE_PATH = os.path.join(MODEL_DIR, "model_structpruned.tflite")
OUT_CC = os.path.join(MODEL_DIR, "val_samples.cc")
OUT_H = os.path.join(MODEL_DIR, "val_samples.h")

N_SAMPLES = 1596

X_val = np.load(os.path.join(DATA_DIR, "val_input.npy")).astype("float32")
y_val = np.load(os.path.join(DATA_DIR, "val_output.npy")).astype("int32")

rng = np.random.default_rng(42)
idx = rng.choice(len(X_val), size=min(N_SAMPLES, len(X_val)), replace=False)
X_sub = X_val[idx]
y_sub = y_val[idx]

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
scale, zero_point = input_details["quantization"]

X_q = np.round(X_sub / scale + zero_point).astype(np.int8)

with open(OUT_CC, "w") as f:
    f.write('#include "val_samples.h"\n\n')
    f.write(f"const int NUM_VAL_SAMPLES = {len(X_q)};\n")
    f.write("const int8_t val_samples[][5] = {\n")
    for row in X_q:
        f.write("  {" + ", ".join(str(v) for v in row) + "},\n")
    f.write("};\n\n")
    f.write("const uint8_t val_labels[] = {\n  ")
    f.write(", ".join(str(v) for v in y_sub))
    f.write("\n};\n")

with open(OUT_H, "w") as f:
    f.write("#ifndef VAL_SAMPLES_H\n#define VAL_SAMPLES_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("extern const int NUM_VAL_SAMPLES;\n")
    f.write("extern const int8_t val_samples[][5];\n")
    f.write("extern const uint8_t val_labels[];\n\n")
    f.write("#endif\n")

print("Saved", OUT_CC, "and", OUT_H, f"with {len(X_q)} samples")
print("Input scale:", scale, "zero_point:", zero_point)
