import numpy as np
import pandas as pd

np.random.seed(42)

N_NORMAL = 6000
N_ANOMALY_EACH = 500

# label map: 0=normal, 1=spike, 2=flooding, 3=replay, 4=spoofed_id
rows = []
prev_val = None
repeat_count = 0

def push(val, interval, msg_id, label):
    global prev_val, repeat_count
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

df.to_csv("dataset.csv", index=False)
print(df.label.value_counts())
print("Saved dataset.csv with", len(df), "rows")
