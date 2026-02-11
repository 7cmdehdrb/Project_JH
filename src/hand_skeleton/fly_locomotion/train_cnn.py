import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# =========================================================
# 1) Load dataset (same format as your txt)
#    line: "<label_str> f1 f2 ... f168"
#    label_str: rotation -> class 0, translation/else -> class 1
# =========================================================
txt_path = "/home/hoon/hand_skeleton/skeleton_dataset/train/train_241201.txt"

skeleton_data = []
labels_int = []
pointing = 0
none = 0

with open(txt_path, "r") as f:
    for line in f.readlines():
        data = line.split()
        skeleton_data.append(data[1:])

        if data[0] == "translation":
            labels_int.append(1)  # class 1
            none += 1
        elif data[0] == "rotation":
            labels_int.append(0)  # class 0
            pointing += 1
        else:
            labels_int.append(1)  # class 1
            none += 1

X = np.array(skeleton_data, dtype=np.float32)
y = np.array(labels_int, dtype=np.int64)

print("X:", X.shape, "y:", y.shape, "pointing/none:", pointing, none)

num_classes = 2
y_onehot = to_categorical(y, num_classes=num_classes)

# =========================================================
# 2) Train/Val split
# =========================================================
X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
    X, y_onehot, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 3) Reshape: (N, 168) -> (N, J, C)
#    Here: C=3 (xyz), J=168/3=56
# =========================================================
input_dim = X_train.shape[1]
if input_dim % 3 != 0:
    raise ValueError(f"input_dim={input_dim} is not divisible by 3. Can't reshape to (J,3).")

J = input_dim // 3  # number of joint tokens
C = 3               # xyz

X_train = X_train.reshape(-1, J, C)
X_val   = X_val.reshape(-1, J, C)

print("Reshaped X_train:", X_train.shape, "X_val:", X_val.shape)

# =========================================================
# 4) Build 1D-CNN (Conv along joint axis)
# =========================================================
def build_1d_cnn(num_classes: int, J: int, C: int, dropout: float = 0.3) -> tf.keras.Model:
    inp = layers.Input(shape=(J, C))  # (56,3)

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out, name="cnn1d_joint")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_1d_cnn(num_classes=num_classes, J=J, C=C, dropout=0.3)
model.summary()

# =========================================================
# 5) Train
# =========================================================
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    # callbacks=[early_stopping],
)

# =========================================================
# 6) Evaluate on validation set (recommended)
# =========================================================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"[1D-CNN] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

pred = model.predict(X_val, verbose=0)
pred_cls = np.argmax(pred, axis=1)

acc = accuracy_score(y_val_int, pred_cls)
prec = precision_score(y_val_int, pred_cls, average="weighted")
rec = recall_score(y_val_int, pred_cls, average="weighted")
f1 = f1_score(y_val_int, pred_cls, average="weighted")
print(f"[1D-CNN] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

# =========================================================
# 7) Plot loss curve
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs", fontsize=12, fontname="Times New Roman")
plt.ylabel("Loss", fontsize=12, fontname="Times New Roman")
plt.grid()
# plt.savefig("/home/hoon/hand_skeleton/training_loss_plot_cnn1d.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional save
model.save("/home/hoon/hand_skeleton/fly_locomotion/model/model_1dcnn.h5")
