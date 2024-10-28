"""
Project : "양궁 개인 최적동작 판별 알고리즘"
Subject : "모델 학습"
Version : 1.1
Started : 2023-10-15
Updated : 2024-10-28
Language: Python
Supervised: Jihoon, Park
"""

import os
from collections import Counter
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from modules import lstm_model as lm
from modules import preprocess as pre

# 커맨드라인 인자 처리
if len(sys.argv) < 2:
    print(
        """        -------------------------------------------------
        선수명 코드를 입력하여 주십시오. python3 model_learning.py {code}
        -------------------------------------------------""")
    sys.exit(1)

player_code = sys.argv[1]

current_dir = os.getcwd()

# 전처리 된 npy 파일 저장경로
npy_directory = os.path.join(current_dir, "sample/npy", player_code)  # 현재는 sample 로 시뮬레이션
# npy_directory = os.path.join(current_dir, "DATA")

# 학습자료 불러오기
x_train = pre.npy_loads(npy_directory, "x_train")
y_train = pre.npy_loads(npy_directory, "y_train")

print(f"x_train sample: {len(x_train)}")
print(f"y_train sample: {len(y_train)}")

# 점수 변경(2차 년도)
old_labels = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
new_labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

label_map = dict(zip(old_labels, new_labels))
y_train = [label_map[label] for label in y_train]

print(Counter(y_train))

# y_train 자료 원핫인코딩
num_classes = max(y_train) + 1
y_train = pre.one_hot_encode_labels(y_train, num_classes)

# array 배열로 변환
x_train = np.array(x_train)
y_train = np.array(y_train)

# 학습 & 테스트 자료 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42, shuffle=True
)

# 빠른 학습을 위해 float32 타입 변경
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

print(f"Training set size: {x_train.shape}, {y_train.shape}")
print(f"Test set size: {x_test.shape}, {y_test.shape}")

# 학습파라미터 설정 및 학습실행
if y_train.size > 0:
    # Train the LSTM model
    history, test_loss, test_acc = lm.Archery_LSTM(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=16,
        max_frame=900,
        num_keypoints=6,
        epoch=15,
        player_code=player_code
    )
else:
    print("훈련 데이터가 없습니다. 데이터 로딩 프로세스를 확인하십시오.")

