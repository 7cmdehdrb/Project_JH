import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

def load_txt_dataset(txt_path: str):
    skeleton_data = []
    labels_onehot = []
    pointing = 0
    none = 0

    with open(txt_path, "r") as f:
        for line in f.readlines():
            data = line.split()
            skeleton_data.append(data[1:])

            if data[0] == "translation":
                labels_onehot.append([0, 1])  # class 1
                none += 1
            elif data[0] == "rotation":
                labels_onehot.append([1, 0])  # class 0
                pointing += 1
            else:
                labels_onehot.append([0, 1])
                none += 1

    X = np.array(skeleton_data, dtype=np.float32)
    y_onehot = np.array(labels_onehot, dtype=np.float32)
    y_int = np.argmax(y_onehot, axis=1)

    return X, y_onehot, y_int, pointing, none

def reshape_for_seq_models(X_flat: np.ndarray):
    # (N,168) -> (N,J,3)
    D = X_flat.shape[1]
    if D % 3 != 0:
        raise ValueError(f"Input dim {D} is not divisible by 3.")
    J = D // 3
    return X_flat.reshape(-1, J, 3)

def _sync_if_gpu():
    try:
        tf.experimental.sync_devices()
    except Exception:
        pass

def stream_infer_and_measure(
    model: tf.keras.Model,
    X_input: np.ndarray,
    y_true_int: np.ndarray,
    warmup: int = 50,
):
    """
    X_input: 이미 모델 입력 shape로 준비된 numpy array
             - MLP: (N,168)
             - CNN/TR: (N,J,3)
    y_true_int: (N,)
    """
    # tf.function으로 고정해서 python overhead를 줄임
    @tf.function(jit_compile=False)
    def forward(x):
        return model(x, training=False)

    # warm-up (앞부분 일부로)
    n = X_input.shape[0]
    w = min(warmup, n)
    for i in range(w):
        x = tf.convert_to_tensor(X_input[i:i+1])
        _ = forward(x)
        _sync_if_gpu()

    # streaming inference (batch=1)
    preds = np.empty((n,), dtype=np.int64)

    t0 = time.perf_counter()
    for i in range(n):
        x = tf.convert_to_tensor(X_input[i:i+1])
        p = forward(x)
        _sync_if_gpu()
        preds[i] = int(tf.argmax(p[0], axis=-1))
    t1 = time.perf_counter()

    total_s = t1 - t0
    ms_per_sample = (total_s / n) * 1000.0
    samples_per_sec = n / total_s

    # metrics
    acc = accuracy_score(y_true_int, preds)
    prec = precision_score(y_true_int, preds, average="weighted", zero_division=0)
    rec = recall_score(y_true_int, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_true_int, preds, average="weighted", zero_division=0)

    return {
        "total_s": total_s,
        "ms_per_sample": ms_per_sample,
        "samples_per_sec": samples_per_sec,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
    }

# =========================================================
# 사용법
# =========================================================
# 1) 네가 이미 학습해 둔 모델 객체를 준비:
#    mlp_model, cnn_model, tr_model
#

@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = models.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout,
        })
        return config

mlp_model = tf.keras.models.load_model("/home/hoon/hand_skeleton/fly_locomotion/model/model_mlp.h5")
cnn_model = tf.keras.models.load_model("/home/hoon/hand_skeleton/fly_locomotion/model/model_1dcnn.h5")
tr_model  = tf.keras.models.load_model("/home/hoon/hand_skeleton/fly_locomotion/model/model_transformer.keras", custom_objects={"TransformerEncoder": TransformerEncoder},)

# 2) test txt 경로 지정
test_txt_path = "/home/hoon/hand_skeleton/skeleton_dataset/evaluate/evaluate_241104.txt"

X_test_flat, y_test_onehot, y_test_int, p_cnt, n_cnt = load_txt_dataset(test_txt_path)
print("Test X:", X_test_flat.shape, "Test y:", y_test_int.shape, "pointing/none:", p_cnt, n_cnt)

# 입력 준비
X_test_mlp = X_test_flat                      # (N,168)
X_test_seq = reshape_for_seq_models(X_test_flat)  # (N,J,3)

# =========================================================
# 3) 각 모델에 대해 streaming latency + test 성능 평가
# =========================================================
results_mlp = stream_infer_and_measure(mlp_model, X_test_mlp, y_test_int)
results_cnn = stream_infer_and_measure(cnn_model, X_test_seq, y_test_int)
results_tr  = stream_infer_and_measure(tr_model,  X_test_seq, y_test_int)

def print_result(name, r):
    print(f"\n[{name}]")
    print(f"Total time      : {r['total_s']:.4f} s")
    print(f"Avg latency     : {r['ms_per_sample']:.4f} ms/sample")
    print(f"Throughput      : {r['samples_per_sec']:.2f} samples/s")
    print(f"Acc / Prec / Rec / F1 : {r['acc']:.8f} / {r['prec']:.8f} / {r['rec']:.8f} / {r['f1']:.8f}")

print_result("MLP", results_mlp)
print_result("1D-CNN", results_cnn)
print_result("Transformer", results_tr)
