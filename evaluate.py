import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
 accuracy_score, precision_score, recall_score, f1_score,
 confusion_matrix
)

from inference import predict

y_true = []
y_pred = []

# Load dataset
with open("dataset.json", "r") as f:
 data = json.load(f)

for item in data:
 text = item.get("text", "")
 video = item.get("video", None)
 label = item["label"]

 result = predict(text=text, video_path=video)
 pred = 1 if result["label"] == "FAKE" else 0

 y_true.append(label)
 y_pred.append(pred)

# ── Metrics ─────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# ── 1. Confusion Matrix ─────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["REAL","FAKE"],
            yticklabels=["REAL","FAKE"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ── 2. Metrics Bar Chart ────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values  = [acc, prec, rec, f1]

plt.figure(figsize=(6,4))
plt.bar(metrics, values)
plt.ylim(0,1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.show()

# ── 3. Accuracy Pie Chart ───────────────────────────────
correct = sum([1 for i in range(len(y_true)) if y_true[i]==y_pred[i]])
incorrect = len(y_true) - correct

plt.figure(figsize=(5,5))
plt.pie([correct, incorrect],
        labels=["Correct","Incorrect"],
        autopct="%1.1f%%",
        colors=["green","red"])
plt.title("Accuracy Distribution")
plt.show()