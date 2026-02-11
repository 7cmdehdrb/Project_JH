import tensorflow as tf

# tf.config.set_visible_devices([], "GPU")
from tensorflow.keras.models import *
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

model = load_model("model/model_mlp_tuned_20260131_175050.h5")

dataset = open(
    "/home/hoon/hand_skeleton/skeleton_dataset/evaluate/evaluate_241104.txt", "r"
)

skeleton_data = []
labels = []

for line in dataset.readlines():
    data = line.split()
    skeleton_data.append(data[1:])
    if data[0] == "translation":
        labels.append([0, 1])
    elif data[0] == "rotation":
        labels.append([1, 0])
    else:
        labels.append([0, 1])

skeleton_data = np.array(skeleton_data).astype(float)
labels = np.array(labels)

loss, accuracy = model.evaluate(skeleton_data, labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

predictions = model.predict(skeleton_data)
predictions = np.argmax(predictions, axis=1)
labels = np.argmax(labels, axis=1)

# print("Classification Report")
# print(classification_report(labels, predictions))
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy}")
precision = precision_score(labels, predictions, average="weighted")
print(f"Precision: {precision}")
recall = recall_score(labels, predictions, average="weighted")
print(f"Recall: {recall}")
f1 = f1_score(labels, predictions, average="weighted")
print(f"F1: {f1}")

cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
