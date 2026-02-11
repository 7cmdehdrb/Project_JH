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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

txt = open("/home/hoon/hand_skeleton/skeleton_dataset/train/train_241201.txt", "r")
skeleton_data = []
labels = []
trans = 0
rot = 0
none = 0

for line in txt.readlines():
    data = line.split()
    skeleton_data.append(data[1:])
    if data[0] == "translation":
        labels.append([1, 0, 0])
        trans += 1
    elif data[0] == "rotation":
        labels.append([0, 1, 0])
        rot += 1
    else:
        labels.append([0, 0, 1])
        none += 1

skeleton_data = np.array(skeleton_data).astype(float)
labels = np.array(labels)

print(skeleton_data.shape)
print(labels.shape)

model = Sequential()
model.add(Dense(128, input_dim=168, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    skeleton_data, labels, epochs=100, batch_size=32, validation_split=0.2
)

loss, accuracy = model.evaluate(skeleton_data, labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

model.save("/home/hoon/hand_skeleton/skeleton_dataset/model/model_train_241201_test.h5")

history_dict = history.history
loss = history_dict["loss"][-1]
val_loss = history_dict["val_loss"][-1]
accuracy = history_dict["accuracy"][-1]
val_accuracy = history_dict["val_accuracy"][-1]

print(f"Loss: {loss}")
print(f"Validation Loss: {val_loss}")
print(f"Accuracy: {accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

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

print(trans, rot, none)
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# predictions = model.predict(new_skeleton_data)
