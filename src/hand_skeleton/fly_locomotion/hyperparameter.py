"""
MLP Hyperparameter Tuning using Optuna
======================================
원본 mlp.py를 기반으로 하이퍼파라미터 튜닝을 수행합니다.

튜닝 대상 파라미터:
- 레이어 수 (1~4)
- 각 레이어의 뉴런 수 (32~256)
- 활성화 함수 (relu, tanh, elu, swish)
- Optimizer (adam, sgd, rmsprop, adamw)
- Learning Rate (1e-5 ~ 1e-2)
- Batch Size (16, 32, 64, 128)
- Dropout Rate (0.0 ~ 0.5)
- L2 정규화 (1e-6 ~ 1e-2)
"""

import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings
import json
from datetime import datetime
import os

warnings.filterwarnings("ignore")

# ============================================================================
# 설정
# ============================================================================
DATA_PATH = "/home/hoon/hand_skeleton/skeleton_dataset/train/train_241201.txt"
MODEL_SAVE_PATH = "/home/hoon/hand_skeleton/fly_locomotion/model"
N_TRIALS = 100 # 튜닝 시도 횟수
TIMEOUT = 360000000000000000000000  # 최대 튜닝 시간 (초)
RANDOM_STATE = 42

# GPU 메모리 제한 설정 (선택사항)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ============================================================================
# 데이터 로드
# ============================================================================
def load_data(data_path):
    """데이터 로드 및 전처리"""
    txt = open(data_path, "r")
    skeleton_data = []
    labels = []

    for line in txt.readlines():
        data = line.split()
        skeleton_data.append(data[1:])
        if data[0] == "translation":
            labels.append([0, 1])  # One-hot 대신 정수 레이블 사용
        elif data[0] == "rotation":
            labels.append([1, 0])
        else:
            labels.append([0, 1])

    txt.close()
    skeleton_data = np.array(skeleton_data).astype(float)
    labels = np.array(labels)

    print(f"데이터 형태: {skeleton_data.shape}")
    print(f"레이블 분포: pointing={sum(labels==[1, 0])}, non{sum(labels==[1, 0])}")

    return skeleton_data, labels


# ============================================================================
# 모델 생성 함수
# ============================================================================
def create_model(trial, input_dim):
    """Optuna trial 기반 모델 생성"""

    # 하이퍼파라미터 샘플링
    n_layers = trial.suggest_int("n_layers", 1, 4)
    activation = "relu"

    model = Sequential()

    # 입력 레이어
    first_units = trial.suggest_int("units_layer_0", 32, 256, step=32)
    model.add(
        Dense(
            first_units,
            input_dim=input_dim,
            activation=activation
        )
    )
    # if use_batch_norm:
    #     model.add(BatchNormalization())
    # if dropout_rate > 0:
    #     model.add(Dropout(dropout_rate))

    # 은닉 레이어
    for i in range(1, n_layers):
        units = trial.suggest_int(f"units_layer_{i}", 32, 256, step=32)
        model.add(Dense(units, activation=activation))
        # if use_batch_norm:
        #     model.add(BatchNormalization())
        # if dropout_rate > 0:
        #     model.add(Dropout(dropout_rate))

    # 출력 레이어
    model.add(Dense(2, activation="softmax"))

    # # Optimizer 설정
    # optimizer_name = trial.suggest_categorical(
    #     "optimizer", ["adam", "sgd", "rmsprop", "adamw"]
    # )
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # if optimizer_name == "adam":
    #     optimizer = Adam(learning_rate=learning_rate)
    # elif optimizer_name == "sgd":
    #     momentum = trial.suggest_float("sgd_momentum", 0.8, 0.99)
    #     optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    # elif optimizer_name == "rmsprop":
    #     optimizer = RMSprop(learning_rate=learning_rate)
    # else:  # adamw
    #     weight_decay = trial.suggest_float("adamw_weight_decay", 1e-5, 1e-2, log=True)
    #     optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# ============================================================================
# 목적 함수 (Optuna)
# ============================================================================
def objective(trial):
    """Optuna 목적 함수 - 교차 검증으로 성능 평가"""

    # 학습 관련 하이퍼파라미터
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = trial.suggest_int("epochs", 50, 200, step=50)
    patience = trial.suggest_int("early_stopping_patience", 10, 30, step=5)

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # 모델 생성
    model = create_model(trial, input_dim=X_train.shape[1])

    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    # 학습
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # 평가
    y_pred = np.argmax(model.predict(X_val), axis=1)
    val_score = f1_score(np.argmax(y_val,axis=1), y_pred, average="weighted")

    # 메모리 정리
    tf.keras.backend.clear_session()

    return val_score


# ============================================================================
# 최적 모델 학습 및 저장
# ============================================================================
def train_best_model(best_params, X_train, y_train, X_val, y_val, input_dim):
    """최적 하이퍼파라미터로 최종 모델 학습"""

    # 모델 구성
    model = Sequential()

    n_layers = best_params["n_layers"]
    activation = "relu"

    # 레이어 추가
    for i in range(n_layers):
        units = best_params[f"units_layer_{i}"]
        if i == 0:
            model.add(
                Dense(
                    units,
                    input_dim=input_dim,
                    activation=activation
                )
            )
        else:
            model.add(
                Dense(units, activation=activation)
            )

    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # 콜백
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=best_params["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 학습
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        callbacks=callbacks,
    )

    return model, history


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MLP 하이퍼파라미터 튜닝 시작")
    print("=" * 60)

    # 데이터 로드
    X_data, y_data = load_data(DATA_PATH)

    # Train/Test 분할 (최종 평가용)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE, stratify=y_data
    )

    # Optuna 스터디 생성
    study = optuna.create_study(
        direction="maximize",  # F1 score 최대화
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )

    # 튜닝 실행
    print(f"\n총 {N_TRIALS}회 시도, 최대 {TIMEOUT}초 동안 튜닝 수행")
    study.optimize(
        objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("튜닝 결과")
    print("=" * 60)
    print(f"최고 F1 Score: {study.best_value:.4f}")
    print(f"시도 횟수: {len(study.trials)}")

    print("\n최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 최적 하이퍼파라미터로 최종 모델 학습
    print("\n" + "=" * 60)
    print("최적 모델 학습")
    print("=" * 60)

    # Train/Val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    best_model, history = train_best_model(
        study.best_params, X_train, y_train, X_val, y_val, X_train.shape[1]
    )

    # 테스트 세트 평가
    print("\n" + "=" * 60)
    print("최종 평가 (테스트 세트)")
    print("=" * 60)

    y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    test_accuracy = accuracy_score(np.argmax(y_test,axis=1), y_pred)
    test_f1 = f1_score(np.argmax(y_test,axis=1), y_pred, average="weighted")

    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 F1 Score: {test_f1:.4f}")

    # 모델 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_mlp_tuned_{timestamp}.h5"
    model_path = os.path.join(MODEL_SAVE_PATH, model_filename)
    best_model.save(model_path)
    print(f"\n모델 저장됨: {model_path}")

    # 하이퍼파라미터 저장
    params_filename = f"best_params_{timestamp}.json"
    params_path = os.path.join(MODEL_SAVE_PATH, params_filename)
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params": study.best_params,
                "best_f1_score": study.best_value,
                "test_accuracy": test_accuracy,
                "test_f1_score": test_f1,
            },
            f,
            indent=2,
        )
    print(f"파라미터 저장됨: {params_path}")

    # 시각화 저장 (선택사항)
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_html(
            os.path.join(MODEL_SAVE_PATH, f"optimization_history_{timestamp}.html")
        )

        fig2 = plot_param_importances(study)
        fig2.write_html(
            os.path.join(MODEL_SAVE_PATH, f"param_importances_{timestamp}.html")
        )

        fig3 = plot_parallel_coordinate(study)
        fig3.write_html(
            os.path.join(MODEL_SAVE_PATH, f"parallel_coordinate_{timestamp}.html")
        )

        print("\n시각화 파일 저장 완료")
    except Exception as e:
        print(f"시각화 저장 중 오류: {e}")

    print("\n" + "=" * 60)
    print("하이퍼파라미터 튜닝 완료!")
    print("=" * 60)
