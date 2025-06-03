# Project/modules/json_preprocess.py
import json
import os
import re

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

"""
사전에 전처리된 json 파일들을 대상으로  
모델 학습 및 시각화를 위한 전처리 코드
"""


# 파일명으로 부터 선수코드 찾아내기
def extract_number_from_filename(filename):
    pattern = r"^(\d{2})_"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return None


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


# 시각화 동영상 처리 함수
def process_video(
    video_path,
    output_path,
    json_keypoints,
    mean_coordinates,
    sample_coordinates,
    score_probs,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return

    # model 예측치
    max_label = max(score_probs, key=score_probs.get)
    predict = {}
    for label in ["Good", "Normal", "Bad"]:
        color = (0, 255, 0) if label == max_label else (255, 255, 255)
        predict[label] = (score_probs[label], color)

    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    selected_keypoints = [5, 6, 7, 8, 9, 10]  # 시각화할 키포인트 인덱스

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_number >= sample_coordinates.shape[0] or frame_number >= len(
            json_keypoints
        ):
            break

        # 현재 프레임의 키포인트 구조 자동 감지
        frame_data = json_keypoints[frame_number]
        keypoint_data = (
            frame_data[0] if isinstance(frame_data[0][0], (list, tuple)) else frame_data
        )

        for i, kp_index in enumerate(selected_keypoints):
            if kp_index >= len(keypoint_data):
                print(f"⚠️ Frame {frame_number}: missing keypoint index {kp_index}")
                continue

            # 평균 및 샘플 좌표
            x_mean = mean_coordinates[frame_number, 2 * i]
            y_mean = mean_coordinates[frame_number, 2 * i + 1]
            x_sample = sample_coordinates[frame_number, 2 * i]
            y_sample = sample_coordinates[frame_number, 2 * i + 1]

            # 차이 비율 계산
            x_diff_ratio = (x_sample - x_mean) / x_mean if x_mean != 0 else 0
            y_diff_ratio = (y_sample - y_mean) / y_mean if y_mean != 0 else 0

            # JSON에서 관절 좌표 추출
            keypoint = keypoint_data[kp_index]
            x_json, y_json, confidence = keypoint  # y, x 순서

            if confidence > 0.3:
                x_json = int(x_json)
                y_json = int(y_json)

                # 색상 선택
                color_x = get_color_by_ratio(x_diff_ratio)
                color_y = get_color_by_ratio(y_diff_ratio)

                # 시각화
                if color_x:
                    cv2.circle(frame, (x_json, y_json), 5, color_x, -1)
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
                    cv2.circle(frame, (x_json, y_json), 5, color_y, -1)
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

                # 마지막 60프레임에 모델 정보 표시
                if frame_number >= total_frames - 60:
                    base_y = 30
                    for idx, (label, (score, color)) in enumerate(predict.items()):
                        text = f"{label}: {score}"
                        cv2.putText(
                            frame,
                            text,
                            (10, base_y + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

        out.write(frame)

        # 영상 실시간 재생
        # cv2.imshow("Frame", frame)
        #
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
