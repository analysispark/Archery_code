# Project/modules/json_preprocess.py

import json
import os
import re

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def convert_keypoints(keypoints):
    if isinstance(keypoints[0][0][0], list):
        return keypoints
    else:
        print("데이터구조 변환")
        return [[keypoint] for keypoint in keypoints]

def dimensional_check(json_path):
    try:
        if not os.path.exists(json_path):
            print(f"File {json_path} does not exist.")
            return  

        with open(json_path, "r") as f:
            data = json.load(f)

                data["annotation"]["2d_keypoints"] = convert_keypoints(
                    data["annotation"]["2d_keypoints"]
                )
            with open(json_path, "w") as f_out:
                    json.dump(data, f_out, indent=4)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

def get_folder_list(directory):
    # 디렉토리 내 항목 목록 가져오기
    items = os.listdir(directory)
    # 숨김 파일과 디렉토리가 아닌 항목 제외
    folders = [
        item
        for item in items
        if not item.startswith(".") and os.path.isdir(os.path.join(directory, item))
    ]
    print(folders)
    return folders


def find_max_frame(json_path):
    # 단일 json 파일로 부터 프레임 길이 계산 (영상 편집을 하였기 때문에 실제 프레임 계산 필요)
    max_frame = -1  # 최대값을 초기화합니다.

    for json_file in json_path:
        try:
            with open(json_file, "r") as file:
                json_data = json.load(file)
                frame = int(len(json_data["annotation"]["2d_keypoints"]))
                if frame > max_frame:
                    max_frame = frame  # 최대값을 업데이트합니다.

        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    return max_frame


def json_max_frame(json_path):
    # 여러 json 파일로부터 프레임 최대값 탐색
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


def extract_number_from_filename(filename):
    # 현재는 score가 없으므로 선수명 코드 추출
    pattern = r"^(\d{2})_"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return None


def gen_y_train(json_files):
    # 현재는 score가 없으므로 선수명 코드 추출
    filename = os.path.basename(json_files)
    label = extract_number_from_filename(filename)
    return label


def transform_json(input_json, max_frame_length=900, keypoints_indices=None):
    output_json = []

    with open(input_json, "r") as file:
        json_data = json.load(file)
        keypoints_data = json_data["annotation"]["2d_keypoints"]

    if keypoints_indices is None:
        keypoints_indices = [
            5,
            6,
            7,
            8,
            9,
            10,
        ]  # 기본 키포인트 인덱스 (left_shoulder 부터 right_wrist 까지)

    num_keypoints = len(keypoints_indices)

    for frame in keypoints_data:
        new_frame = []
        for keypoints in frame:
            # 지정된 키포인트 인덱스만 추출
            new_keypoints = [
                [point[0], point[1]]
                for i, point in enumerate(keypoints)
                if i in keypoints_indices
            ]
            new_frame.extend(new_keypoints)
        if new_frame:
            flattened_data = np.array(new_frame).reshape(-1)
            output_json.append(flattened_data)
        else:
            print(f"Warning: Empty frame data in file {input_json}")

    # 데이터를 (None, 2 * num_keypoints) 형태로 변환
    output_json = np.array(output_json)

    # MinMaxScaler를 사용하여 데이터를 정규화합니다.
    scaler = MinMaxScaler()
    output_json_reshaped = output_json.reshape(-1, num_keypoints * 2)
    output_json_normalized = scaler.fit_transform(output_json_reshaped)
    output_json_normalized = output_json_normalized.reshape(-1, num_keypoints * 2)

    # 입력 데이터에 따라 빈 프레임 추가
    if len(output_json_normalized) < max_frame_length:
        padding_length = max_frame_length - len(output_json_normalized)
        padding_frames = np.zeros((padding_length, num_keypoints * 2))
        output_json_normalized = np.concatenate(
            (output_json_normalized, padding_frames), axis=0
        )

    # 최종 데이터를 (None, 1821, 2 * num_keypoints) 형태로 변환
    output_json_normalized = output_json_normalized.reshape(
        -1, max_frame_length, num_keypoints * 2
    )

    return output_json_normalized


def process_files_in_folder(current_dir, folder_list, max_frame, Output_path):
    for i in folder_list:
        json_files = search_json_files(os.path.join(current_dir, "DATA/Json", i), 900)

        x_train = []
        y_train = []

        for json_file in json_files:
            try:
                processed_data = transform_json(json_file, max_frame)

                if len(processed_data) > 0:
                    x_train.append(processed_data)
                    y_train.append(gen_y_train(json_file))
                else:
                    print(
                        f"No valid data or score in file: {json_file}"
                    )  # 현재는 score가 없음. 버전2에서 score 사용
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")

        # 리스트를 numpy 배열로 변환
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], -1))

        # 파일 저장
        np.save(os.path.join(Output_path, f"x_train_{i}.npy"), x_train)
        np.save(os.path.join(Output_path, f"y_train_{i}.npy"), y_train)
