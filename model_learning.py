"""
Project : "양궁 개인 최적동작 판별 알고리즘"
Subject : "모델 학습"
Version : 2.1
Started : 2023-10-15
Updated : 2025-06-02
Language: Python
Supervised: Jihoon, Park
"""

import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

import modules

# 커맨드라인 인자 처리
if len(sys.argv) < 2:
    print(
        """        -------------------------------------------------
        선수명 코드를 입력하여 주십시오. python3 model_learning.py {code}
        -------------------------------------------------"""
    )
    sys.exit(1)

player_code = sys.argv[1]

####### 06.03
Data_path = os.path.join(os.getcwd(), "Data")
Record_path = os.path.join(Data_path, "record.json")
Json_path = os.path.join(Data_path, "Jsons")
Video_path = os.path.join(Data_path, "Videos")
Npy_path = os.path.join(Data_path, "npy")
Vis_path = os.path.join(Data_path, "visual_temp")


x_train, y_train = modules.load_npy(Npy_path, player_code)

# 점수- Gold 변경
old_labels = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
new_labels = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2]

label_map = dict(zip(old_labels, new_labels))
y_train = [label_map[label] for label in y_train]

print(Counter(y_train))

# y_train 자료 원핫인코딩
y_train = modules.one_hot_encode_labels(y_train, 3)

# array 배열로 변환
x_train = np.array(x_train)
y_train = np.array(y_train)

# 학습 & 테스트 자료 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42, shuffle=True
)


results = {}

# 학습파라미터 설정 및 학습실행
if y_train.size > 10:
    # Train the BiGRU model
    model, history, test_loss, test_acc = modules.train_or_finetune_archery_model(
        player_code=player_code,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    print(f"[{player_code}] 선수 슈팅 최적동작 학습완료.")
else:
    print(f"[{player_code}] 선수의 슈팅자료가 부족합니다. {y_train.size}/1,000건")
