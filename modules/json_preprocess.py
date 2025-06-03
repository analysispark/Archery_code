# Project/modules/json_preprocess.py

import json
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

"""
원본 json으로 부터 데이터구조 변환 및 표준화 전처리 코드 모음
"""


def one_hot_encode_labels(labels, num_classes):
    labels_array = np.array(labels)  # 정수 리스트를 numpy 배열로 변환
    return to_categorical(labels_array, num_classes=num_classes)


def load_npy(npy_path, number):
    print(f"npy_path: {npy_path}")
    print(os.path.join(npy_path, f"x_train_{number}.npy"))
    x_npy = np.load(os.path.join(npy_path, f"x_train_{number}.npy"))
    y_npy = np.load(os.path.join(npy_path, f"y_train_{number}.npy"))
    print(f"Loaded array shape: x_train = {x_npy.shape}")
    print(f"Loaded array shape: y_train = {y_npy.shape}")
    return x_npy, y_npy


# 폴더 내 항목 리스트 담는 코드
def get_folder_list(directory):
    # 디렉토리 내 항목 목록 가져오기
    items = os.listdir(directory)
    # 숨김 파일과 디렉토리가 아닌 항목 제외
    folders = [
        item
        for item in items
        if not item.startswith(".") and os.path.isdir(os.path.join(directory, item))
    ]
    return folders


# 여러 json 파일로부터 프레임 최대값 탐색
def json_max_frame(json_path):
    # 임시 파일은 예외 처리
    if os.path.basename(json_path).startswith("._"):
        print(f"Skipping temporary file: {json_path}")
        return -1  # -1을 반환하여 임시 파일을 처리하지 않도록 함

    max_frame = -1  # 최대값을 초기화합니다.

    try:
        with open(json_path, "r") as file:
            json_data = json.load(file)
            max_frame = int(len(json_data["annotation"]["2d_keypoints"]))
    except Exception as e:
        print(f"Error processing file {json_path}: {e}")

    return max_frame


# 폴더 내에서 json 파일 탐색 코드
def search_json_files(directory, frame):
    json_files = []  # JSON 파일을 저장할 리스트

    for root, dirs, files in os.walk(directory):
        # 현재 디렉토리의 파일들 중 .json 확장자를 가진 파일들을 리스트에 추가
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                max_frame = json_max_frame(json_path)
                if (
                    max_frame != -1 and max_frame <= frame
                ):  # 설정 프레임 이하인 파일만 추가
                    json_files.append(json_path)
                else:
                    print(f"Excluding file {json_path} with {max_frame} frames")

        # 하위 폴더가 있다면, 재귀적으로 해당 폴더들에 대해 검색 수행
        for dir in dirs:
            json_files.extend(search_json_files(os.path.join(root, dir), frame))
        break  # os.walk()의 첫 번째 레벨만 사용하고, 나머지는 재귀 호출을 통해 처리
    return json_files


def extract_score(json_file):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            score = data.get("info", {}).get("score", None)

            if score is None or score == "None":
                return 0
            else:
                return int(score)  # 필요에 따라 float(score) 가능
    except Exception as e:
        print(f"Error extracting score from {json_file}: {e}")
        return 0


def transform_json(input_json, max_frame_length=900, keypoints_indices=None):
    output_json = []

    with open(input_json, "r") as file:
        json_data = json.load(file)
        keypoints_data = json_data["annotation"]["2d_keypoints"]

    if keypoints_indices is None:
        keypoints_indices = [5, 6, 7, 8, 9, 10]  # left_shoulder ~ right_wrist

    num_keypoints = len(keypoints_indices)

    for frame in keypoints_data:
        new_frame = []
        for i in keypoints_indices:
            if i < len(frame):
                x, y, c = frame[i]
                new_frame.extend([x, y])  # confidence 제외
            else:
                new_frame.extend([0, 0])  # 누락된 키포인트는 0으로 패딩
        output_json.append(np.array(new_frame))

    output_json = np.array(output_json)

    # 정규화
    scaler = MinMaxScaler()
    output_json_normalized = scaler.fit_transform(output_json)

    # 패딩
    if len(output_json_normalized) < max_frame_length:
        padding_len = max_frame_length - len(output_json_normalized)
        padding = np.zeros((padding_len, num_keypoints * 2))
        output_json_normalized = np.concatenate(
            [output_json_normalized, padding], axis=0
        )
    else:
        output_json_normalized = output_json_normalized[:max_frame_length]

    # 최종 반환 형태
    return output_json_normalized.reshape(1, max_frame_length, num_keypoints * 2)


# 원본 json 파일들을 리스트에 담고, 정규화&전처리 실행 및 npy 변환하여 저장한는 실행코드
def process_files_in_folder(json_dir, folder_list, max_frame, Output_path):
    for i in folder_list:
        json_files = search_json_files(os.path.join(json_dir, i), max_frame)

        x_train = []
        y_train = []

        for json_file in json_files:
            try:
                score = extract_score(json_file)
                if score < 5:
                    print(f"Score < 5, skipping file: {json_file}")
                    continue

                processed_data = transform_json(json_file, max_frame)

                # 빈 데이터 처리
                if processed_data is None or len(processed_data) == 0:
                    print(f"Empty or invalid processed data in file: {json_file}")
                    continue

                if processed_data.shape[0] == 0:
                    print(f"Empty shape in processed data: {json_file}")
                    continue

                x_train.append(processed_data)
                y_train.append(score)

            except Exception as e:
                print(f"Error processing file {json_file}: {e}")

        if len(x_train) == 0:
            print(f"No data to save for folder {i}. Skipping save.")
            continue

        # 리스트를 numpy 배열로 변환
        x_train = np.array(x_train)
        y_train = np.array(y_train, dtype=np.int32)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], -1))

        # 점수- Gold 변경
        old_labels = [10, 9, 8, 7, 6, 5]
        new_labels = [0, 0, 1, 1, 2, 2]

        label_map = dict(zip(old_labels, new_labels))
        y_train = [label_map[label] for label in y_train]

        # y_train 자료 원핫인코딩
        y_train = modules.one_hot_encode_labels(y_train, 3)

        # 파일 저장
        np.save(os.path.join(Output_path, f"x_train_{i}.npy"), x_train)
        np.save(os.path.join(Output_path, f"y_train_{i}.npy"), y_train)
