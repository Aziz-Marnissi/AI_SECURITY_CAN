# SecureAI_IDS — TinyML Intrusion Detection for CAN Bus (ESP32)

A lightweight, edge-deployed AI system that detects anomalies and attacks on an automotive CAN bus network, using a quantized and structurally pruned neural network running on an ESP32 microcontroller.

---

## 1. System Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  CAN Bus Traffic │ --> │  Feature Extraction   │ --> │  TFLite Micro    │
│  (or simulated    │     │  (value, interval,    │     │  MLP Classifier  │
│   dataset)         │     │   id, delta,          │     │  on ESP32        │
│                    │     │   repeat_count)        │     │                  │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                                                              v
                                                    ┌─────────────────────┐
                                                    │  5-class prediction  │
                                                    │  Normal / Spike /     │
                                                    │  Flooding / Replay /  │
                                                    │  Spoofed_ID           │
                                                    └─────────────────────┘
```

**Pipeline stages:**
1. **Data generation** — synthetic CAN bus traffic simulating normal operation and 4 attack types
2. **Training** — a small MLP (multilayer perceptron) trained to classify traffic into 5 classes
3. **Pruning** — structural pruning to shrink the network's neuron count
4. **Quantization** — converting float32 weights to int8 for embedded deployment
5. **Deployment** — the quantized model is compiled into a C array and run on an ESP32 via TensorFlow Lite Micro (TFLite Micro)
6. **On-device validation** — real hardware testing of accuracy, RAM, flash usage, and inference latency

---

## 2. The Dataset

Since real CAN bus attack datasets are scarce and hardware-specific, a synthetic dataset was generated (`generate_dataset.py`) simulating realistic CAN traffic patterns:

| Feature | Meaning |
|---|---|
| `value` | signal value carried by the CAN message (e.g. sensor reading) |
| `interval_ms` | time since the last message on this bus (ms) |
| `id` | CAN arbitration ID (which ECU/message type sent it) |
| `delta` | absolute change in `value` since the previous message |
| `repeat_count` | how many consecutive messages have (nearly) the same value |

### Classes simulated
| Label | Class | Description |
|---|---|---|
| 0 | **Normal** | baseline sensor traffic, slowly drifting values around fixed setpoints, regular ~100ms intervals |
| 1 | **Spike** | value jumps to an implausible extreme (sensor fault or injected data) |
| 2 | **Flooding** | messages arrive far more frequently than normal (denial-of-service style attack) |
| 3 | **Replay** | the exact same value is repeated many times in a row (a captured message being replayed) |
| 4 | **Spoofed_ID** | a message uses an arbitration ID that shouldn't exist on this bus (impersonating another ECU) |

Final dataset: **7,980 rows** (6,000 normal + ~500 per attack type), split 80/20 for train/test with stratification (`train_test_split(..., stratify=y)`), so each class is proportionally represented in both sets.

---

## 3. The AI Model — Theory

### 3.1 Why a simple MLP (not CNN/RNN)?
CAN bus features here are tabular (5 independent numeric features per message), not sequential/spatial data — a small **dense feedforward network (MLP)** is the right tool: cheap to compute, tiny memory footprint, and sufficient accuracy for this feature set. CNNs/RNNs would add compute cost with no benefit here.

### 3.2 Baseline architecture
```
Input(5) → Dense(16, relu) → Dense(8, relu) → Dense(5, softmax)
```
- **ReLU** activations in hidden layers (cheap, avoids vanishing gradients)
- **Softmax** output for 5-class probability distribution
- Trained with `sparse_categorical_crossentropy` loss, Adam optimizer, early stopping on validation loss

**Baseline result: 99.00% test accuracy.**

### 3.3 Why compress the model at all?
The ESP32 has ~320KB RAM and ~1.3MB flash shared with the whole firmware (WiFi stack, CAN driver, application logic). Even a tiny model must be as cheap as possible in **RAM (working memory during inference)** and **inference latency (time to classify one message)**, since CAN buses can carry thousands of messages per second.

### 3.4 Quantization (float32 → int8)
Quantization maps the model's float32 weights/activations to int8 using a `scale` and `zero_point` per tensor:
```
real_value = (int8_value - zero_point) * scale
```
This is **post-training integer quantization** using a representative dataset (300 real samples) so the converter can calibrate realistic min/max ranges per layer. Benefits:
- theoretically ~4x smaller tensors (32-bit → 8-bit)
- faster integer arithmetic on microcontrollers without an FPU
- required format for `TFLITE_BUILTINS_INT8` ops on ESP32 via TFLite Micro

### 3.5 Pruning — two approaches compared

**(a) Unstructured (magnitude) pruning** — via `tensorflow_model_optimization` (`tfmot`). Individual weights below a magnitude threshold are zeroed out, following a polynomial sparsity schedule (0% → 60% sparsity over training). This keeps the architecture shape (16→8) unchanged, just with many zeroed connections.

> **Key finding**: TFLite's default flatbuffer format stores every weight *including zeros* — sparsity alone does **not** shrink the `.tflite` file. It only helps with **compression** (e.g. `.gz` for OTA transfer: our model compressed from 3.41KB → 1.52KB, a 55% reduction) and can help RAM/latency slightly via zero-skipping in some kernels.

**(b) Structural pruning (neuron removal)** — the theoretically correct technique for shrinking microcontroller models. Instead of zeroing individual weights, entire neurons are removed:
1. Rank each hidden-layer neuron by the **L1 norm** of its weights + bias (a proxy for how much it contributes to the network's output)
2. Keep only the top-k neurons per layer, discard the rest
3. **Transplant** the kept neurons' weights into a smaller network (16→8 becomes 8→4)
4. **Fine-tune** the smaller network briefly to recover accuracy lost from the size reduction

This actually shrinks the tensors themselves, so the `.tflite` file, RAM usage, and compute all shrink for real — no compression tricks needed.

**Fine-tuning recovery**: pre-fine-tune accuracy after neuron removal was 89.47%; after 20 epochs of fine-tuning, it recovered to 98.93% (final quantized structural model: 98.62–99.00% depending on eval set).

---

## 4. Results — Host-side Evaluation (all 3 model variants)

| Metric | Baseline (fp32, 16-8) | Unstructured pruned+quant (int8, 16-8, 53.8% sparse) | Structural pruned+quant (int8, 8-4) |
|---|---|---|---|
| Model size (.tflite) | 3.21 KB | 3.41 KB (+6.2%, quantization metadata overhead) | **3.01 KB (−6.2%)** |
| Model size (gzip) | — | 1.52 KB (−55.4% vs raw) | — |
| Estimated RAM (tensor arena) | 1.23 KB | 0.39 KB (−68%) | **0.18 KB (−85%)** |
| Host CPU latency (proxy only) | 0.0028 ms | 0.0028 ms | 0.0029 ms |
| Test accuracy | 99.00% | 98.68% | 98.62% |

**Conclusion**: unstructured pruning only helps if you compress for storage/OTA transfer. Structural pruning is the only method that reduces real, on-device flash and RAM — this is why it was chosen for final ESP32 deployment.

---

## 5. Results — Real ESP32 Hardware (ground truth)

Host-side numbers above are **proxies only** (desktop TFLite interpreter with XNNPACK acceleration — not representative of a microcontroller). Real numbers were captured by flashing each model onto an ESP32 with TensorFlow Lite Micro (`Chirale_TensorFLowLite` PlatformIO library) and running the full 1,596-sample test set on-device.

| Metric | Baseline (fp32, 16-8) | Structural pruned+quant (int8, 8-4) | Change |
|---|---|---|---|
| Flash (full firmware incl. framework) | 596,445 B (45.5%) | 572,285 B (43.7%) | −4.1% |
| RAM (framework) | 33,624 B (10.3%) | 31,624 B (9.7%) | −5.9% |
| Tensor arena used (model only) | — | 964 B | — |
| **On-device accuracy (1,596 samples)** | **99.00%** | **98.37%** | −0.63 pts |
| **Avg inference latency (real hardware)** | **71.22 µs** | **56.45 µs** | **−20.7%** |
| Free heap after init | 333,144 B | 335,144 B | +2,000 B |

**Bottom line**: structural pruning + quantization trades a small accuracy drop (0.63 points) for a meaningful 20.7% inference speedup and lower RAM footprint — a solid tradeoff for a real-time IDS running alongside other ESP32 workloads (CAN bus polling, WiFi, etc.).

---

## 6. Plots Explained

### `plots/plot_loss.png` and `plots/plot_accuracy.png`
Training curves over 80 epochs (baseline model), with early stopping. Train and validation curves track closely together with no divergence — confirms the model is **not overfitting**. Both converge within ~5-10 epochs; remaining epochs show minor fluctuation, not real improvement.

### `plots/plot_confusion_matrix.png`
Row = true class, column = predicted class. Diagonal = correct predictions. The only notable off-diagonal cluster is **13 Replay samples misclassified as Normal** — expected, because a replayed message (a real value repeated) is statistically indistinguishable from a normal reading unless the model specifically learns the `repeat_count` feature pattern. All other classes (Spike, Flooding, Spoofed_ID) are classified with zero confusion.

### `plots/plot_roc.png`
One-vs-rest ROC curves per class. All classes achieve AUC ≥ 0.994 — Replay is the weakest (0.994) matching the confusion matrix finding, all others reach 1.000 or near-1.000.

### `plots/compare_baseline_vs_structural.png`
Four-panel comparison (host-side estimates): model size, estimated RAM, host CPU latency, and accuracy for baseline vs. structural pruned+quantized. This is the **host proxy** version — see Section 5 for the real, ground-truth ESP32 numbers.

---

## 7. Project Structure

```
SecureAI_IDS_L476RG/
├── data/
│   ├── dataset.csv              # generated synthetic CAN traffic (7,980 rows)
│   ├── norm_mean.npy             # per-feature mean (for input normalization)
│   ├── norm_std.npy              # per-feature std (for input normalization)
│   ├── val_input.npy              # held-out test set inputs
│   └── val_output.npy             # held-out test set labels
├── models/
│   ├── anomaly_model.h5              # baseline trained Keras model
│   ├── anomaly_model_pruned.h5        # unstructured (magnitude) pruned model
│   ├── anomaly_model_structpruned.h5   # structurally pruned + fine-tuned model
│   ├── model_baseline.tflite            # baseline, fp32, unquantized
│   ├── model.tflite                       # unstructured pruned + int8 quantized
│   ├── model_structpruned.tflite           # structural pruned + int8 quantized (deployed)
│   ├── model_data.cc / .h                   # C array of model_structpruned.tflite (ESP32)
│   ├── model_data_baseline.cc / .h            # C array of baseline model (for comparison)
│   ├── model_data_unstruct.cc / .h             # C array of unstructured pruned model
│   └── val_samples*.cc / .h                      # embedded validation samples for on-device testing
├── plots/                                          # all generated plots (see Section 6)
├── generate_dataset.py                               # Step 1: synthetic dataset generation
├── train_model.py                                      # baseline training (standalone script)
├── run_pipeline.py                                       # full pipeline: generate → train → prune → export → quantize
├── structured_prune.py                                     # structural (neuron-level) pruning + fine-tuning
├── compare_models.py / compare_all.py / compare_baseline_vs_structural.py
│                                                              # host-side comparison scripts (size/RAM/latency/accuracy)
├── export_struct_c_array.py                                    # regenerate C array from structural model
├── export_baseline_c_array.py                                     # regenerate C array from baseline model
├── export_unstructured_c_array.py                                  # regenerate C array from unstructured pruned model
├── export_val_samples_c.py                                           # export validation samples as C array (structural model's scale)
└── check_pipeline.sh                                                    # sanity-check script (verifies pipeline ran correctly)
```

### ESP32 deployment projects (separate PlatformIO folders)
```
~/esp32_ids_test/            # structural pruned+quantized model — final deployment target
~/esp32_ids_baseline/        # baseline fp32 model — for comparison
~/esp32_ids_unstructured/    # unstructured pruned+quantized model — for comparison
```
Each is a standalone PlatformIO project with:
- `platformio.ini` — ESP32 board config + `Chirale_TensorFLowLite` library dependency
- `src/main.cpp` — loads the model, runs inference over embedded validation samples, prints accuracy/latency/RAM to Serial
- `src/model_data*.cc`, `include/model_data*.h` — the model as a C byte array
- `src/val_samples*.cc`, `include/val_samples*.h` — quantized validation samples + true labels

---

## 8. Setup & Reproduction

### 8.1 Python environment (training/pruning/quantization)
```bash
pip install --break-system-packages tensorflow tensorflow-model-optimization scikit-learn pandas matplotlib
```

### 8.2 Run the full pipeline
```bash
cd SecureAI_IDS_L476RG
python3 run_pipeline.py          # generate data, train baseline, unstructured-prune, quantize, export C array
python3 structured_prune.py      # structural pruning + fine-tuning
python3 export_struct_c_array.py # regenerate C array from the structural model
```

### 8.3 Compare all variants (host-side)
```bash
python3 compare_all.py
```

### 8.4 ESP32 deployment (PlatformIO)
```bash
pip install --break-system-packages platformio
mkdir -p esp32_ids_test && cd esp32_ids_test
platformio project init --board esp32dev --project-option "framework=arduino"
```
Add to `platformio.ini`:
```ini
lib_deps =
    spaziochirale/Chirale_TensorFLowLite
build_flags =
    -DBOARD_HAS_PSRAM
```
Copy the model C array and validation samples into `src/`/`include/`, write `main.cpp` (see `~/esp32_ids_test/src/main.cpp` in this repo), then:
```bash
platformio run --target upload
platformio device monitor --baud 115200
```

### 8.5 Sanity check
```bash
bash check_pipeline.sh
```
Verifies the pipeline script contains pruning code, that all expected model/data files exist, and that the pruned model was actually regenerated after the base model (catches stale-script bugs).

---

## 9. Key Takeaways

1. **Unstructured (magnitude) pruning does not reduce on-device flash/RAM** unless paired with sparse storage or compression — a common misconception. It's still useful for OTA transfer size (gzip) and can marginally help latency.
2. **Structural pruning (neuron removal) is the correct technique for microcontroller deployment** — it physically shrinks the model's tensors, giving real reductions in flash, RAM, and compute.
3. **Fine-tuning after structural pruning is essential** — accuracy dropped to 89.47% immediately after neuron removal, but recovered to ~99% after 20 epochs of fine-tuning.
4. **Host-side benchmarks are proxies only** — always validate on real target hardware. Host CPU latency (with XNNPACK) was ~20x faster than real ESP32 numbers and told a misleadingly flat story; only on-device testing revealed the true 20.7% latency improvement from pruning.
5. **This is a CAN bus IDS proof-of-concept** using synthetic data — a production system would need real vehicle CAN bus captures, additional attack patterns, and integration with actual CAN transceiver hardware on the ESP32.
