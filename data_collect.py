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
from pathlib import Path

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


# 선수별 Json 폴더번호 리스트로 저장
folder_list = modules.get_folder_list(Json_path)

# npy 저장 폴더가 없을 경우 생성
if not os.path.exists(Npy_path):
    os.makedirs(Npy_path)


# # 선수별 원본 json 파일들을 리스트에 담고, 정규화 & 전처리 실행 및 npy 변환
# print("정규화 & 전처리 실행 및 npy 변환 완료")

processed_files = modules.load_previous_record(Record_path)
all_files = modules.scan_all_json_files(Path(Json_path))

new_files = [f for f in all_files if f not in processed_files]

if not new_files:
    print("✅ 새로운 파일 없음.")
else:
    print(f"🆕 새로운 파일 {len(new_files)}개 발견!")
    folder_list = [f.name for f in Path(Json_path).iterdir() if f.is_dir()]
    modules.process_files_in_folder(Json_path, folder_list, 900, Npy_path)

for rel_path in new_files:
    folder_id, file_name = rel_path.split(os.sep)
    file_path = Path(Json_path) / folder_id / file_name
    processed_files.add(rel_path)

modules.save_current_record(processed_files, Record_path)
