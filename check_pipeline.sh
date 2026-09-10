#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OK=1

check() {
    local label="$1" cond="$2" hint="$3"
    if [ "$cond" -eq 0 ]; then
        echo "[OK ] $label"
    else
        echo "[FAIL] $label"
        [ -n "$hint" ] && echo "       -> $hint"
        OK=0
    fi
}

if [ -f "$ROOT/run_pipeline.py" ]; then
    grep -q "def prune(" "$ROOT/run_pipeline.py"
    check "run_pipeline.py has prune() function" $? "Outdated script — replace it."
    grep -q "prune(X_norm, y)" "$ROOT/run_pipeline.py"
    check "main block calls prune(X_norm, y)" $? "prune() never called."
    grep -q "anomaly_model_pruned.h5" "$ROOT/run_pipeline.py"
    check "tflite conversion uses pruned model" $? "convert_tflite() points at base model."
else
    echo "[FAIL] run_pipeline.py not found"
    OK=0
fi

for f in "data/dataset.csv" "models/anomaly_model.h5" "models/anomaly_model_pruned.h5" \
         "models/model.tflite" "models/model_data.cc"; do
    [ -f "$ROOT/$f" ]
    check "$f exists" $? "Missing — rerun: python3 run_pipeline.py"
done

if [ -f "$ROOT/models/anomaly_model.h5" ] && [ -f "$ROOT/models/anomaly_model_pruned.h5" ]; then
    [ "$ROOT/models/anomaly_model_pruned.h5" -nt "$ROOT/models/anomaly_model.h5" ]
    check "pruned model created AFTER base model" $? "pruning step didn't rerun."
fi

echo
if [ "$OK" -eq 1 ]; then echo "All checks passed."; else echo "Issues found. Rerun pipeline."; exit 1; fi
