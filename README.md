

## 프로젝트 개요
---

Project : "양궁 개인 최적동작 분석"  
Subject : "모델 학습"  
Version : 2.0  
Started : 2023-10-15  
Updated : 2025-05-30  
Language: Python  
Supervised: Jihoon, Park  

---

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
│   ├── x_train_46.npy
│   ├── y_train_46.npy 
│   └── ...
│
└── visual_temp # 시각화 영상이 임시로 저장되는 폴더 (재생 뒤에는 영상 삭제 ; 영구저장할 것인지 삭제할 것인지는 플랫폼에서 판단)
    └── 46_999_091122_01_vis.mp4


**Data folder list path**
Data_path = os.path.join(os.getcwd(), "Data")
Record_path = os.path.join(Data_path, "record.json")
Json_path = os.path.join(Data_path, "Jsons")
Video_path = os.path.join(Data_path, "Videos")
Npy_path = os.path.join(Data_path, "npy")
Vis_path = os.path.join(Data_path, "visual_temp")


---
국가대표 양궁선수들의 슈팅영상에서 HPE(Human Pose Estimation)를 이용하여 관절포인트를 추출하고 동작의 일관성을 판단하는 모델 생성    
구체적으로 **(1)관절포인트를 추출한 Json 파일을 학습모델 자료로 생성**하는 모듈과   
**(2)생성된 학습 자료세트를 활용하여 양궁선수 슈팅동작의 일관성을 판단하는 모델** 생성    
  
  
  
  
  
**프로세스**  (2025-06-04일 기준)

- 각 프로세스는 독립적임
- 0번 부터 순차적으로 종속적 관계 (예: 3번 프로세스는 0~2까지가 수행되어야 실행 가능)
  
``` bash
0. 선수 슈팅동작 촬영 및 자료 누적
1. 누적된 다수 json자료 단일 npy로 스택을 쌓아 저장(슈팅동작 관절변화와 라벨링(스코어) 자료 총 2개 npy)
2. npy 파일을 사용하여 개인 슈팅동작 일관성 모델학습 및 모델의 가중치 저장
3. 원본 json, npy(슈팅동작 npy; 스코어 npy는 필요없음), 원본 동영상을 사용하여 시각화
```
    
  
** Todo **

- 2차년도 누적영상 최종 선정(불필요 & 이상 영상 삭제)
- 스코어 y_train.npy 생성 코드 (스위트케이와 협의 필요; 추후 스코어를 어디에 기록할 것인지 확인(현재는 슈팅동작 json 파일의 score 저장으로 가정))
- 샘플 생성 및 시뮬레이션 코드 작성
- 시각화 부분 고도화(프레임별 오차표시 외 학습모델의 종합판정 결과값을 표기하도록 추가 예정)


## Contents

1. [설치](#설치)
2. [코드 실행](#코드-실행)
3. [학습자료 생성 프로세스](#학습자료-생성-프로세스)
4. [모델학습 프로세스](#모델학습-프로세스)
5. [시각화 프로세스](#시각화-프로세스)

---

## 1. 설치

#### 요구 사양
1. CUDA (optional)
2. tensorflow 
3. python3

#### Linux & WSL2

```bash
git clone https://github.com/analysispark/Project_Archery.git
cd Project_Archery
pip install -r requirements.txt
sh get_DATA.sh
```



## 2. 코드 실행

**< 학습 자료 생성 >**
```bash
python data_collect.py
```

**< 모델 학습 >**

```bash
python model_learning.py
```

**< 시각화 >**
```bash
python visualization.py {video_file_name}
```

## 3. 학습자료 생성 프로세스

**modules**    
└── **json_preprocess.py**

**Json parsing:**
- 디렉토리 내 항목 목록 가져오기
- 단일 json 파일로 부터 프레임 길이 계산
- 여러 json 파일로부터 프레임 최대값 탐색
- json 확장자를 가진 파일들을 리스트에 추가하고 최대 프레임을 설정
- label 설정을 위한 파일로 부터 선수 코드명 추출
- json 파일구조로 부터 데이터 형태변환
  - json 파일 로드
  - 관절포인트 설정(left_shoulder ~ right_wrist 까지가 기본값)
  - 각 프레임별 관절포인트 추출
  - 데이터를 (None, 2 * num_keypoints) 형태로 변환
  - 프레임별 관절포인트들의 데이터 정규화
  - 설정 최대프레임에 부족한 만큼 빈 프레임 추가
  - 최종 데이터를 (None, 900(기본값), 2*num_keypoints) 형태로 변환

- 형태변환된 json 자료들을 불러들여 데이터(x_train)에 누적시키면서 4차원 형태로 변환 및 `npy` 확장자로 저장

\* **형태변환한 뒤 json 자료를 불러들여 누적작업** 하는 이유는 각 슈팅동작(`1개의 json 파일`)에서 **프레임별로 정규화**가 이루어져야 하기 때문(`영상 별로 카메라와 선수들의 거리, 프레임 내에서의 선수위치가 다름`)  
\* **각 json 파일별로 정규화 및 데이터 형태를 1차 변환**한 뒤, **자료를 누적하여 4차원 텐서 형태로 최종 변환**하여 학습 데이터 셋 생성




## 4. 모델학습 프로세스

**modules**    
├── **preprocess.py**    
└── **lstm_model.py**    
    
    
    
**Model-learning**:    
- npy 폴더 위치 지정
- 데이터 로드
- labels 재지정
- one-hot-encoding
- 학습 & 테스트 셋 분리
- y_train 데이터 길이 확인
- 모델 학습
  - batch_size = 16
  - max_frame = 900
  - num_keypoints = 6
  - epoch = 30 
  - learning_late = 0.001
  - loss = categorical_crossentropy
- 학습모델 저장 
    

## 5. 시각화 프로세스

**modules**    
├── **extract_landmarks.py**    
├── **preprocess.py**    
└── **json_preprocess.py**    

- 시각화를 위해서는 **원본 json** 파일과 **npy** 파일, **원본 동영상** 파일 3개가 필요
- **원본 json** 파일은 각 프레임별 오차값을 표기할 위치를 찾기 위함
- **npy** 파일은 누적된 선수 슈팅동작에 따른 오차값을 계산하기 위함
- **원본 동영상** 파일은 npy 파일과 원본 json에서 계산된 값을 동영상 위에 얹어야 함
- 총 3개의 파일이 짝으로 이루어져 있어야 시각화 가능함
