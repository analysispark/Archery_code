import os
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

"""
양방향 LSTM 모델 구성 및 추가 학습 기능 추가
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
        # 기존 모델 불러오기
        model = keras.models.load_model(model_path)
        print(f"Loaded existing model from {model_path}")
        
        # 추가 학습을 위해 컴파일은 생략하거나 필요시 optimizer를 다시 설정
        print(f"Fine-tuning the loaded model")
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

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",  # 정수형 레이블에 맞는 손실 함수
            metrics=["accuracy"],
        )

    model.summary()

    # 추가 학습 진행
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

