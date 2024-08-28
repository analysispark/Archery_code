"""
Project : "양궁 개인 최적동작 판별 알고리즘"
Subject : "Json 파일 전처리 및 학습자료 변환"
Version : 1.1
Started : 2023-10-15
Updated : 2024-08-08
Language: Python
Supervised: Jihoon, Park
"""

import os

from modules import json_preprocess as jp

current_dir = os.getcwd()

# Json 파일위치
json_path = os.path.join(current_dir, "DATA")
Output_path = os.path.join(current_dir, "DATA/npy")
folder_list = jp.get_folder_list(os.path.join(json_path, "Json"))

# 900 frame 이하 파일만 저장
total_json_files = jp.search_json_files(json_path, frame=900)  # frame(900)
max_frame = jp.find_max_frame(total_json_files)

# Output_path 폴더가 없을 경우 생성
if not os.path.exists(Output_path):
    os.makedirs(Output_path)

# 선수명코드(파일이름)로 부터 y_train 라벨링 및 x_train tensor 생성
jp.process_files_in_folder(current_dir, folder_list, max_frame, Output_path)
