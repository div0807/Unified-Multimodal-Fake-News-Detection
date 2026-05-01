import pandas as pd
import os
import random
import json

# -------------------------
# LOAD TEXT DATA
# -------------------------
real1 = pd.read_csv("datasets/politifact_real.csv")
fake1 = pd.read_csv("datasets/politifact_fake.csv")
real2 = pd.read_csv("datasets/gossipcop_real.csv")
fake2 = pd.read_csv("datasets/gossipcop_fake.csv")

real1["label"] = 0
real2["label"] = 0
fake1["label"] = 1
fake2["label"] = 1

data = pd.concat([real1, fake1, real2, fake2], ignore_index=True)
data = data[['title', 'label']].dropna()
data.rename(columns={"title": "text"}, inplace=True)
data = data.sample(frac=0.4, random_state=42).reset_index(drop=True)

# -------------------------
# LOAD VIDEO DATA
# -------------------------
base_path = "datasets/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"

videos = []
for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith(".mp4"):
            # Always store with forward slashes — works on Windows, Linux, and Mac
            path = os.path.join(root, f)
            videos.append(path.replace("\\", "/"))

print("Total videos:", len(videos))

# LIMIT DATA
videos = random.sample(videos, 200)

# -------------------------
# CREATE DATASET
# -------------------------
samples = []
min_len = min(len(data), len(videos))

for i in range(min_len):
    samples.append({
        "text":  data.loc[i, "text"],
        "video": videos[i],
        "label": int(data.loc[i, "label"])
    })

with open("dataset.json", "w") as f:
    json.dump(samples, f, indent=2)

print("FINAL SAMPLES:", len(samples))