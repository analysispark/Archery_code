import glob
import os

import numpy as np


def randomize_all_y_train_labels(npy_dir, prefix="y_train_"):
    npy_files = glob.glob(os.path.join(npy_dir, f"{prefix}*.npy"))

    for npy_file in npy_files:
        y_train = np.load(npy_file)
        length = y_train.shape[0]

        # 0~10 정수 범위로 무작위 생성
        new_y = np.random.randint(0, 11, size=length, dtype=np.int32)

        np.save(npy_file, new_y)
        print(f"{os.path.basename(npy_file)}: Replaced with random integers (0~10).")


# 사용 예
npy_path = "/home/charles/Archery_code/Data/npy"
randomize_all_y_train_labels(npy_path)
