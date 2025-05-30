"""
Project : "양궁 개인 최적동작 분석"
Subject : "모델 학습"
Version : 2.0
Started : 2023-10-15
Updated : 2025-05-30
Language: Python
Supervised: Jihoon, Park
"""

# 1. 선수 훈련 영상 수집
# 2. 데이터베이스 확인(신규선수 일 경우 생성, 기존일 경우 불러오기)
# 3. 수집된 영상 전처리 및 패턴 자료 누적
# 3.1. 최소 자료 누적 시(유효 영상 1,000발) 최적동작 모델 학습 및 가중치 생성
# 3.2. 선수 개인의 최적동작 가중치 모델 및 누적자료 저장
# 4. 플랫폼에서 "분석" 버튼 활성화
# 5. "분석" 버튼 클릭으로 선택한 영상의 슈팅분석이 진행되고, 시각화 제공

"""
폴더 구조는 다음과 같다고 가정하에 작성되었음

-- 플랫폼 구조에 맞게 스위트케이에서 변경 --


Data
├── Jsons       # 스위트케이 관절추출된 json이 저장되는 경로
│   ├── 46
│   │   ├── 46_999_091122_00.json
│   │   ├── 46_999_091122_01.json
│   │   ├── ...
│   │   └── 46_999_095403_04.json
│   │── ...
│   └── 50
│
├── Videos      # 스위트케이 선수 영상이 저장되는 경로 
│   ├── 46
│   │   ├── 46_999_091122_00.mp4
│   │   ├── 46_999_091122_01.mp4
│   │   ├── ...
│   │   └── 46_999_095403_04.mp4
│   │── ...
│   └── 50
│
├── npy         # 용인대 코드에서 생성되는 경로(수집된 영상 전처리 및 패턴 자료 누적 경로)
│   ├── 46
│   │   ├── x_train_46.npy
│   │   └── y_train_46.npy
│   ├── ... 
│   └── 50
│
└── visual_temp # 시각화 영상이 임시로 저장되는 폴더 (재생 뒤에는 영상 삭제 ; 영구저장할 것인지 삭제할 것인지는 플랫폼에서 판단)
    └── 46_999_091122_01_vis.mp4

"""
import json
import os
import sys

# Data folder list path
Data_path = os.getcwd()
Json_path = os.path.join(Data_path, "Jsons")
Video_path = os.path.join(Data_path, "Videos")
Npy_path = os.path.join(Data_path, "npy")
Vis_path = os.path.join(Data_path, "visual_temp")

import modules

# 신규영상 체크
modules.json_update_checker()
