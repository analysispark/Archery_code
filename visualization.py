"""
Project : "양궁 개인 최적동작 판별 알고리즘"
Subject : "결과 시각화"
Version : 2.1
Started : 2023-12-04
Updated : 2025-06-03
Language: Python
Supervised: Jihoon, Park
"""

import os
import sys

import numpy as np
import tensorflow as tf

import modules

# Data folder list path
Data_path = os.path.join(os.getcwd(), "Data")
Model_path = os.path.join("models")
Json_path = os.path.join(Data_path, "Jsons")
Video_path = os.path.join(Data_path, "Videos")
Vis_path = os.path.join(Data_path, "visual_temp")


if len(sys.argv) != 2:
    print("파일명을 다시 한번 확인하시기 바랍니다.")
else:
    argv = sys.argv[1]
    print(f"동영상: {argv}")

# 파일명으로 부터 선수코드 불러오기
CODE = modules.extract_number_from_filename(argv)
TARGET_FILE = os.path.splitext(os.path.basename(argv))[0]


# Npy_path
Npy_path = os.path.join(Data_path, "npy")

target_json = os.path.join(Json_path, f"{CODE}/{TARGET_FILE}.json")
target_video = os.path.join(Video_path, f"{CODE}/{TARGET_FILE}.mp4")
npy_DATA = np.load(os.path.join(Npy_path, f"x_train_{CODE}.npy"))


mean_coordinates = modules.calculate_mean_coordinates(npy_DATA)
target_DATA = np.array(modules.transform_json(target_json))
reshape_target_DATA = np.reshape(target_DATA, (900, 12))
json_keypoints = modules.load_2d_keypoints(
    target_json
)  # 수치가 표시될 관절포인트 위치 불러오기


# 모델 예측
model = tf.keras.models.load_model(os.path.join(Model_path, f"{CODE}_model.keras"))
predict = model.predict(target_DATA)
score_probs = {}
labels = ["Good", "Normal", "Bad"]

for idx, prob in enumerate(predict[0]):
    score_label = labels[idx]
    score_probs[score_label] = round(prob * 100, 2)


output_path = os.path.join(Vis_path, f"{TARGET_FILE}_vis.mp4")
modules.process_video(
    target_video,
    output_path,
    json_keypoints,
    mean_coordinates,
    reshape_target_DATA,
    score_probs,
)


print(f"{output_path} 에 시각화 영상이 생성되었습니다.")
