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

# tf.config.set_visible_devices([], "GPU")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

txt = open(
    "/home/min/7cmdehdrb/fuck_flight/src/hand_skeleton/skeleton_dataset/train/train.txt",
    "r",
)
skeleton_data = []
labels = []
pointing = 0
none = 0

for line in txt.readlines():
    data = line.split()
    skeleton_data.append(data[1:])
    if data[0] == "translation":
        labels.append([0, 1])
        none += 1
    elif data[0] == "rotation":
        labels.append([1, 0])
        pointing += 1
    else:
        labels.append([0, 1])
        none += 1

skeleton_data = np.array(skeleton_data).astype(float)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    skeleton_data, labels, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Dense(32, input_dim=168, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss", patience=25, restore_best_weights=True
)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
)

loss, accuracy = model.evaluate(skeleton_data, labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

model.save("local_model_gpu.h5")

history_dict = history.history
loss = history_dict["loss"][-1]
val_loss = history_dict["val_loss"][-1]
accuracy = history_dict["accuracy"][-1]
val_accuracy = history_dict["val_accuracy"][-1]

print(f"Loss: {loss}")
print(f"Validation Loss: {val_loss}")
print(f"Accuracy: {accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs", fontsize=12, fontname="Times New Roman")
plt.ylabel("Loss", fontsize=12, fontname="Times New Roman")
plt.grid()
plt.show()

predictions = model.predict(skeleton_data)
predictions = np.argmax(predictions, axis=1)
labels = np.argmax(labels, axis=1)

accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="weighted")
recall = recall_score(labels, predictions, average="weighted")
f1 = f1_score(labels, predictions, average="weighted")

print(f"Precision: {precision}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print(pointing, none)
