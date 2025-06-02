"""
Project : "양궁 개인 최적동작 판별 알고리즘"
Subject : "Json 파일 전처리 및 학습자료 변환"
Version : 2.1
Started : 2023-10-15
Updated : 2025-05-27
Language: Python
Supervised: Jihoon, Park
"""

import os

import modules

Data_path = os.path.join(os.getcwd(), "Data")
Record_path = os.path.join(Data_path, "record.json")
Json_path = os.path.join(Data_path, "Jsons")
Npy_path = os.path.join(Data_path, "npy")

"""
신규영상이 있는지 체크하고 있을 경우 Json 파일로부터 표준화된 npy 형태로 변환하는 실행 파일

실시간으로 감지하고 학습하는것이 아닌, 일정 간격으로 실행한다는 전제이기 때문에 모든 선수(json) 대상

json 파일자료 저장경로의 형태가 "Data/Jsons" 폴더안에 각 선수코드명의 폴더가 존재하고,
선수코드명 폴더 안에 json 파일이 있다고 가정함.
예시) '현대백화점 양궁선수단 유수정 선수 코드번호: 49'
      = Data/Jsons/49/49_999_161117_00.json

전처리된 npy 파일은 "Data/npy" 경로에 선수코드명 이름으로 저장
예시) '유수정 선수'
      = Data/npy/49_x_train.npy
        DATA/npy/49_y_train.npy
"""

# 신규영상 체크
modules.check_and_process_new_files(Record_path, Json_path)

# 선수별 Json 폴더번호 리스트로 저장
folder_list = modules.get_folder_list(Json_path)

# 필요없을 수 있음 - 25.06.03
# # 900 frame 이하 파일만 저장
# total_json_files = modules.search_json_files(json_path, frame=900)  # frame(900)
# max_frame = modules.find_max_frame(total_json_files)


# npy 저장 폴더가 없을 경우 생성
if not os.path.exists(Npy_path):
    os.makedirs(Npy_path)

# 각 슈팅파일별 스코어 점수로 y_train 라벨링  - json의 'score' 로 가정

# 필요없을 수 있음 - 25.06.03
# # json 구조 확인
# for i in total_json_files:
#     jp.dimensional_check(i)
#
# jp.process_files_in_folder(current_dir, folder_list, 900, Output_path)  # 1차년도


# 선수별 원본 json 파일들을 리스트에 담고, 정규화 & 전처리 실행 및 npy 변환
print("최종 마지막 줄 시작")
modules.process_files_in_folder(Json_path, folder_list, 900, Npy_path)
print("마지막줄 완료")
