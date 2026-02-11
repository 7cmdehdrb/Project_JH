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
    "/home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/dataskeleton_data2.txt",
    "r",
)
skeleton_data = []
labels = []
none = 0
one = 0
two = 0

skipped_lines = 0
for line_num, line in enumerate(txt.readlines(), start=1):
    line = line.strip()
    if not line:  # 빈 라인 건너뛰기
        skipped_lines += 1
        continue

    data = line.split()

    # 레이블 + 168개 데이터 = 총 169개 요소 확인
    if len(data) != 169:
        print(f"Line {line_num}: Expected 169 elements, got {len(data)}. Skipping...")
        skipped_lines += 1
        continue

    # 레이블 검증
    if data[0] not in ["else", "one", "two"]:
        print(f"Line {line_num}: Unknown label '{data[0]}'. Skipping...")
        skipped_lines += 1
        continue

    try:
        # 숫자 데이터 변환 시도
        skeleton_values = [float(x) for x in data[1:]]
        skeleton_data.append(skeleton_values)

        if data[0] == "else":
            labels.append([1, 0, 0])
            none += 1
        elif data[0] == "one":
            labels.append([0, 1, 0])
            one += 1
        elif data[0] == "two":
            labels.append([0, 0, 1])
            two += 1
    except ValueError as e:
        print(f"Line {line_num}: Error converting to float: {e}. Skipping...")
        skipped_lines += 1
        continue

print(f"Total lines loaded: {len(skeleton_data)}, Skipped: {skipped_lines}")

skeleton_data = np.array(skeleton_data).astype(float)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    skeleton_data, labels, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Dense(128, input_dim=168, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="softmax"))

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

model.save("local_model_gpu_v2_2.h5")

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

print(none, one, two)
