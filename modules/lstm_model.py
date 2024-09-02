import time
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def park_LSTM(x_train, y_train, x_test, y_test, batch_size=1, max_frame=900, num_keypoints=6, epoch=30):
    num_classes = y_train.shape[1]

    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(512, return_sequences=True, activation="tanh"), input_shape=(max_frame, num_keypoints * 2)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(256, return_sequences=True, activation="tanh")),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False, activation="tanh")),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("models/best_model.h5", save_best_only=True)
    ]

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1, callbacks=callbacks)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    model.save("models/Archery_model.h5")

    return history, test_loss, test_acc

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