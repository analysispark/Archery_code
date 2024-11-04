"""
Project : "양궁 개인 최적동작 판별 알고리즘"
Subject : "결과 시각화"
Version : 1.1
Started : 2023-12-04
Updated : 2024-11-04
Language: Python
Supervised: Jihoon, Park
"""

import os
import sys

import numpy as np

from modules import json_preprocess as jp
from modules import preprocess as pre

current_dir = os.getcwd()

if len(sys.argv) != 2:
    print("파일명을 다시 한번 확인하시기 바랍니다.")
else:
    target_file = sys.argv[1]
    print(f"동영상: {target_file}")

CODE = jp.extract_number_from_filename(target_file)  # 선수코드 불러오기

# target_directory = os.path.join(os.getcwd(), "DATA") # 현재는 sample로 시뮬레이션. 실제로는 DATA로 사용
target_directory = os.path.join(os.getcwd(), "sample")  # npy files 경로
target_json = os.path.join(
    target_directory, f"json/{target_file}.json"
)  # 비디오파일과 동일한 json
target_video = os.path.join(target_directory, f"videos/{target_file}.mp4")
npy_DATA = np.load(os.path.join(target_directory, f"npy/x_train_{CODE}.npy"))

mean_coordinates = pre.calculate_mean_coordinates(npy_DATA)
target_DATA = np.array(jp.transform_json(target_json))
target_DATA = np.reshape(target_DATA, (900, 12))
json_keypoints = pre.load_2d_keypoints(
    target_json
)  # 수치가 표시될 관절포인트 위치 불러오기
output_path = os.path.join(target_directory, f"{target_file}_output.mp4")
pre.process_video(
    target_video, output_path, json_keypoints, mean_coordinates, target_DATA
)
