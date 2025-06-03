import os

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def archery_biGRU(input_shape):
    model = keras.Sequential(
        [
            layers.Bidirectional(
                layers.GRU(128, return_sequences=True, activation="tanh"),
                input_shape=input_shape,
            ),
            layers.Dropout(0.3),
            layers.Bidirectional(
                layers.GRU(32, return_sequences=True, activation="tanh")
            ),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    return model


def train_or_finetune_archery_model(
    x_train,
    y_train,
    x_test,
    y_test,
    player_code,
    batch_size=16,
    epoch=None,
    learning_rate=None,
    model_dir="models",
):

    # 선수별 모델 경로
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{player_code}_model.keras")

    # 신규 학습인지 기존 이어 학습인지 판별
    first_training = not os.path.exists(model_path)

    if first_training:
        print(f"[{player_code}] 새 모델 생성")
        input_shape = (900, 6 * 2)  #  900(frame) * 6(keypoints) * 2(chanel)
        model = archery_biGRU(input_shape)
        epoch = 30 if epoch is None else epoch
        learning_rate = 0.1 if learning_rate is None else learning_rate
    else:
        print(f"[{player_code}] 기존 모델 불러오기: {model_path}")
        model = keras.models.load_model(model_path)

        epoch = 20 if epoch is None else epoch
        learning_rate = 0.0005 if learning_rate is None else learning_rate

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]

    print(f"[{player_code}] 학습 시작 (epoch={epoch}, lr={learning_rate})")
    history = model.fit(
        x_train,
        y_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    return model, history, loss, acc
