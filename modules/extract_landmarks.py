# Project/modules/extract_landmarks.py

import numpy as np


def extract_landmarks(frame):
    landmarks_data = frame.get("landmarks", [])
    x_values = [landmark.get("x", 0.0) for landmark in landmarks_data]
    y_values = [landmark.get("y", 0.0) for landmark in landmarks_data]
    return np.column_stack((x_values, y_values))


def extract_keypoints(frame):
    # "2d_keypoints" 데이터 가져옴. 데이터가 없을 경우 빈 리스트를 반환
    keypoints_data = frame.get("2d_keypoints", [[]])
    # 사용할 관절 키포인트의 인덱스 지정
    # ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    # 의 인덱스는 0부터 10까지
    valid_keypoints_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # 선택된 관절 키포인트의 [y, x] 값만 추출
    # 여기서 keypoints_data[0]는 첫 번째 프레임의 관절 키포인트 데이터를 의미
    # sweet_k 제공 json [y, x, confidence] 순서
    keypoints = [keypoints_data[0][i][:2] for i in valid_keypoints_indices]  # [y, x] 형태로 변경하고, confidence 제외
    
    # Numpy 배열로 변환. 이때 각 관절 키포인트는 [y, x] 형태를 유지
    return np.array(keypoints)
