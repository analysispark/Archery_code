import time
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

"""
양방향 LSTM 모델 구성
"""
def Archery_LSTM(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    max_frame=900,
    num_keypoints=6,
    epoch=30,
    player_code=00,
):
    sm = SMOTE(random_state=15)
    samples, timesteps, features = x_train.shape
    x_train_flat = x_train.reshape(samples, timesteps * features)
    x_train_res, y_train_res = sm.fit_resample(x_train_flat, y_train)
    x_train_res = x_train_res.reshape(-1, timesteps, features)

    # 고유한 클래스 수 계산
    num_classes = len(np.unique(y_train_res))  # y_train_res의 고유한 클래스 수

    model_path = f"models/Archery_model_{player_code}.h5"
    
    # 모델이 존재하는지 확인
    if os.path.exists(model_path):
        # 모델 불러오기
        model = keras.models.load_model(model_path)
        print(f"Loaded existing model from {model_path}")
    else:
        # 새 모델 생성
        model = keras.Sequential(
            [
                layers.Bidirectional(
                    layers.LSTM(512, return_sequences=True, activation="tanh"),
                    input_shape=(max_frame, num_keypoints * 2),
                ),
                layers.Dropout(0.3),
                layers.Bidirectional(
                    layers.LSTM(256, return_sequences=True, activation="tanh")
                ),
                layers.Dropout(0.3),
                layers.Bidirectional(
                    layers.LSTM(128, return_sequences=True, activation="tanh")
                ),
                layers.Dropout(0.3),
                layers.Bidirectional(
                    layers.LSTM(64, return_sequences=False, activation="tanh")
                ),
                layers.Dense(num_classes, activation="softmax"),  # 고유 클래스 수에 따라 수정
            ]
        )
        print("Created a new model")

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",  # 정수형 레이블에 맞는 손실 함수
        metrics=["accuracy"],
    )

    start_time = time.time()
    history = model.fit(
        x_train_res,
        y_train_res,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.1,
    )
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    y_pred = model.predict(x_train_res)

    cm = confusion_matrix(y_train_res, y_pred.argmax(axis=1))

    # 모델 저장
    model.save(model_path)

    return history, test_loss, test_acc




# 사용안함(결과보고서 및 논문에서 모델 정확도를 위한 그래프)
def make_plot(history):
    plt.style.use("fivethirtyeight")

    # 훈련 손실 그래프
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    # 훈련 정확도 그래프
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"], loc="lower right")
    plt.show()
