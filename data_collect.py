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

"""
Json 파일로부터 표준화된 npy 형태로 변환하는 실행 파일


json 파일자료 저장경로의 형태가 "DATA" 폴더안에 각 선수코드명의 폴더가 존재하고,
선수코드명 폴더 안에 json 파일이 있다고 가정함.
예시) '현대백화점 양궁선수단 유수정 선수 코드번호: 49'
      = DATA/49/49_999_161117_00.json

전처리된 npy 파일은 "DATA/npy" 경로에 선수코드명 이름으로 저장
예시) '유수정 선수'
      = DATA/npy/49_x_train.npy
        DATA/npy/49_y_train.npy
"""
# Json 파일위치 경로
json_path = os.path.join(current_dir, "DATA")  # 전체 선수들 json 원본 저장 경로
Output_path = os.path.join(current_dir, "DATA/npy")  # 전처리 완료된 npy 파일 저장 경로
folder_list = jp.get_folder_list(
    os.path.join(json_path, "Json")
)  # 원본 json 저장위치의 선수폴더명 리스트업


# 900 frame 이하 파일만 저장
total_json_files = jp.search_json_files(json_path, frame=900)  # frame(900)
max_frame = jp.find_max_frame(total_json_files)

# Output_path 폴더가 없을 경우 생성
if not os.path.exists(Output_path):
    os.makedirs(Output_path)

# (1차년도) 선수명코드(파일이름)로 부터 y_train 라벨링 및 x_train tensor 생성
# (2차년도) 각 슈팅파일별 스코어 점수로 y_train 라벨링  - json의 'score' 로 가정
jp.process_files_in_folder(current_dir, folder_list, max_frame, Output_path)  # 1차년도
