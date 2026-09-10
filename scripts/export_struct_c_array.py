"""
Regenerate model_data.cc/.h from model_structpruned.tflite (structurally pruned + quantized).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

TFLITE_PATH = os.path.join(MODEL_DIR, "model_structpruned.tflite")
CC_PATH = os.path.join(MODEL_DIR, "model_data.cc")
H_PATH = os.path.join(MODEL_DIR, "model_data.h")

with open(TFLITE_PATH, "rb") as f:
    tflite_model = f.read()

with open(CC_PATH, "w") as f:
    f.write("#include \"model_data.h\"\n\n")
    f.write("alignas(8) const unsigned char g_model[] = {\n")
    for i, b in enumerate(tflite_model):
        f.write(f"0x{b:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n")

with open(H_PATH, "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
    f.write("extern const unsigned char g_model[];\n")
    f.write("extern const unsigned int g_model_len;\n\n")
    f.write("#endif\n")

print("Saved", CC_PATH, f"({len(tflite_model)} bytes)")
print("Saved", H_PATH)
