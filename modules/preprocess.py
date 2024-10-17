# Project/modules/json_preprocess.py

import json
import os

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

"""
사전에 전처리된 json 파일들을 대상으로  
모델 학습 및 시각화를 위한 전처리 코드
"""

# npy 파일 불러오는 함수
def npy_loads(target_directory, prefix):
    npy_files = []

    # 디렉토리 내 파일들 탐색
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file.startswith(prefix) and file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))

    # npy 파일 병합
    arrays = []
    for npy_file in npy_files:
        print(npy_file)
        array = np.load(npy_file)
        arrays.append(array)

    # 병합
    if arrays:
        concatenated_array = np.concatenate(arrays, axis=0)
        return concatenated_array
    else:
        return np.array([])

# 정수값을 원-핫 인코딩 하는 함수
def one_hot_encode_labels(labels, num_classes):
    labels_array = np.array(labels)  # 정수 리스트를 numpy 배열로 변환
    return to_categorical(labels_array, num_classes=num_classes)

# 여러 영상 동일 프레임 관절포인트 평균 계산
def calculate_mean_coordinates(x_train):
    return np.mean(x_train, axis=0)


# 좌표 차이 계산 및 비율 변환
def calculate_coordinate_differences(sample_data, mean_coordinates):
    differences = sample_data - mean_coordinates
    differences_ratio = differences / mean_coordinates
    return differences_ratio


# 2D 키포인트 데이터를 JSON에서 로드 (시각화 위치좌표를 위한 원본 json파일 좌표값 불러오기)
def load_2d_keypoints(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    keypoints = data["annotation"]["2d_keypoints"]
    return keypoints


# 차이 비율에 따른 색상 선택
def get_color_by_ratio(ratio):
    if abs(ratio) <= 0.4:
        return None  # 시각화하지 않음
    elif abs(ratio) <= 0.6:
        return (0, 255, 0)  # 초록색
    else:
        return (0, 0, 255)  # 빨간색


# 시각화 동영상 처리
def process_video(
    video_path, output_path, json_keypoints, mean_coordinates, sample_coordinates
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    selected_keypoints = [5, 6, 7, 8, 9, 10]  # 시각화할 키포인트 인덱스

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_number >= sample_coordinates.shape[0] or frame_number >= len(
            json_keypoints
        ):
            break

        # 각 프레임에서 관절 좌표와 차이 비율을 표시
        keypoint_data = json_keypoints[frame_number][
            0
        ]  # 각 프레임의 첫 번째 인덱스는 키포인트 데이터 리스트
        for i, kp_index in enumerate(selected_keypoints):
            x_mean = mean_coordinates[frame_number, 2 * i]
            y_mean = mean_coordinates[frame_number, 2 * i + 1]
            x_sample = sample_coordinates[frame_number, 2 * i]
            y_sample = sample_coordinates[frame_number, 2 * i + 1]

            # 차이 비율 계산
            x_diff_ratio = (x_sample - x_mean) / x_mean if x_mean != 0 else 0
            y_diff_ratio = (y_sample - y_mean) / y_mean if y_mean != 0 else 0

            # JSON 파일에서 2D 키포인트 좌표 가져오기
            keypoint = keypoint_data[kp_index]
            y_json, x_json, confidence = keypoint  # y, x 순서

            if confidence > 0.5:  # 신뢰도 기준 필터링
                # 좌표를 정수형으로 변환하여 시각화
                x_json = int(x_json)
                y_json = int(y_json)

                # 차이 비율에 따른 색상 선택
                color_x = get_color_by_ratio(x_diff_ratio)
                color_y = get_color_by_ratio(y_diff_ratio)

                # 관절 좌표와 차이 비율을 프레임에 표시
                if color_x:
                    cv2.circle(
                        frame, (x_json, y_json), 5, color_x, -1
                    )  # x 좌표 차이 비율에 따른 색상
                    cv2.putText(
                        frame,
                        f"{x_diff_ratio:.2f}",
                        (x_json, y_json - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_x,
                        1,
                        cv2.LINE_AA,
                    )

                if color_y:
                    cv2.circle(
                        frame, (x_json, y_json), 5, color_y, -1
                    )  # y 좌표 차이 비율에 따른 색상
                    cv2.putText(
                        frame,
                        f"{y_diff_ratio:.2f}",
                        (x_json, y_json - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_y,
                        1,
                        cv2.LINE_AA,
                    )

        out.write(frame)  # 프레임 저장
        cv2.imshow("Frame", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
