import time

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def park_RNN(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    max_frame=900,
    num_keypoints=6,
    epoch=20,
):
    num_classes = y_train.shape[1]

    model = keras.Sequential(
        [
            layers.SimpleRNN(
                128,
                return_sequences=True,
                activation="tanh",
                input_shape=(max_frame, num_keypoints * 2),
            ),
            layers.Dropout(0.3),
            layers.SimpleRNN(32, return_sequences=True, activation="tanh"),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return train_and_evaluate(
        model, x_train, y_train, x_test, y_test, batch_size, epoch
    )


def park_GRU(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    max_frame=900,
    num_keypoints=6,
    epoch=20,
):
    num_classes = y_train.shape[1]

    model = keras.Sequential(
        [
            layers.GRU(
                128,
                return_sequences=True,
                activation="tanh",
                input_shape=(max_frame, num_keypoints * 2),
            ),
            layers.Dropout(0.3),
            layers.GRU(32, return_sequences=True, activation="tanh"),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return train_and_evaluate(
        model, x_train, y_train, x_test, y_test, batch_size, epoch
    )


def park_biGRU(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    max_frame=900,
    num_keypoints=6,
    epoch=20,
):
    num_classes = y_train.shape[1]

    model = keras.Sequential(
        [
            layers.Bidirectional(
                layers.GRU(128, return_sequences=True, activation="tanh"),
                input_shape=(max_frame, num_keypoints * 2),
            ),
            layers.Dropout(0.3),
            layers.Bidirectional(
                layers.GRU(32, return_sequences=True, activation="tanh")
            ),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return train_and_evaluate(
        model, x_train, y_train, x_test, y_test, batch_size, epoch
    )


def park_LSTM_single(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    max_frame=900,
    num_keypoints=6,
    epoch=20,
):
    num_classes = y_train.shape[1]

    model = keras.Sequential(
        [
            layers.LSTM(
                128,
                return_sequences=True,
                activation="tanh",
                input_shape=(max_frame, num_keypoints * 2),
            ),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=True, activation="tanh"),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return train_and_evaluate(
        model, x_train, y_train, x_test, y_test, batch_size, epoch
    )


def park_BiLSTM_single(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    max_frame=900,
    num_keypoints=6,
    epoch=20,
):
    num_classes = y_train.shape[1]

    model = keras.Sequential(
        [
            # 첫 번째 양방향 LSTM 층
            layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, activation="tanh"),
                input_shape=(max_frame, num_keypoints * 2),
            ),
            layers.Dropout(0.3),
            # 두 번째 양방향 LSTM 층
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=True, activation="tanh")
            ),
            layers.Dropout(0.3),
            # 나머지 층들
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return train_and_evaluate(
        model, x_train, y_train, x_test, y_test, batch_size, epoch
    )


def train_and_evaluate(model, x_train, y_train, x_test, y_test, batch_size, epoch):
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("models/best_model.keras", save_best_only=True),
    ]

    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
    )
    end_time = time.time()

    training_time = end_time - start_time
    print(f"훈련 시간: {training_time} 초")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"테스트 정확도: {test_acc*100:.2f}%")

    model.save("models/Archery_model.h5")
    print("훈련 완료")
    return model, history, test_loss, test_acc


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


# import os
# import time
# from collections import Counter
#
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow import keras
# from tensorflow.keras import layers
#
# """
# 양방향 LSTM 모델 구성 및 추가 학습 기능 추가
# """
# def Archery_LSTM(
#     x_train,
#     y_train,
#     x_test,
#     y_test,
#     batch_size=1,
#     max_frame=900,
#     num_keypoints=6,
#     epoch=30,
#     player_code=00,
# ):
#     sm = SMOTE(random_state=15)
#     samples, timesteps, features = x_train.shape
#     x_train_flat = x_train.reshape(samples, timesteps * features)
#     x_train_res, y_train_res = sm.fit_resample(x_train_flat, y_train)
#     x_train_res = x_train_res.reshape(-1, timesteps, features)
#
#     # 고유한 클래스 수 계산
#     num_classes = len(np.unique(y_train_res))  # y_train_res의 고유한 클래스 수
#
#     model_path = f"models/Archery_model_{player_code}.h5"
#
#     # 모델이 존재하는지 확인
#     if os.path.exists(model_path):
#         # 기존 모델 불러오기
#         model = keras.models.load_model(model_path)
#         print(f"Loaded existing model from {model_path}")
#
#         # 추가 학습을 위해 컴파일은 생략하거나 필요시 optimizer를 다시 설정
#         print(f"Fine-tuning the loaded model")
#     else:
#         # 새 모델 생성
#         model = keras.Sequential(
#             [
#                 layers.Bidirectional(
#                     layers.LSTM(512, return_sequences=True, activation="tanh"),
#                     input_shape=(max_frame, num_keypoints * 2),
#                 ),
#                 layers.Dropout(0.3),
#                 layers.Bidirectional(
#                     layers.LSTM(256, return_sequences=True, activation="tanh")
#                 ),
#                 layers.Dropout(0.3),
#                 layers.Bidirectional(
#                     layers.LSTM(128, return_sequences=True, activation="tanh")
#                 ),
#                 layers.Dropout(0.3),
#                 layers.Bidirectional(
#                     layers.LSTM(64, return_sequences=False, activation="tanh")
#                 ),
#                 layers.Dense(num_classes, activation="softmax"),  # 고유 클래스 수에 따라 수정
#             ]
#         )
#         print("Created a new model")
#
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss="sparse_categorical_crossentropy",  # 정수형 레이블에 맞는 손실 함수
#             metrics=["accuracy"],
#         )
#
#     model.summary()
#
#     # 추가 학습 진행
#     start_time = time.time()
#     history = model.fit(
#         x_train_res,
#         y_train_res,
#         epochs=epoch,
#         batch_size=batch_size,
#         validation_split=0.1,
#     )
#     end_time = time.time()
#
#     training_time = end_time - start_time
#     print(f"Training time: {training_time} seconds")
#
#     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
#
#     y_pred = model.predict(x_train_res)
#
#     cm = confusion_matrix(y_train_res, y_pred.argmax(axis=1))
#
#     # 모델 저장
#     model.save(model_path)
#
#     return history, test_loss, test_acc
#
