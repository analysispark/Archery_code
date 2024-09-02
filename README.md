

## 프로젝트 개요

국가대표 양궁선수들의 슈팅영상에서 HPE(Human Pose Estimation)를 이용하여 관절포인트를 추출하고 동작의 일관성을 판단하는 모델 생성    
구체적으로 **(1)관절포인트를 추출한 Json 파일을 학습모델 자료로 생성**하는 모듈과   
**(2)생성된 학습 자료세트를 활용하여 양궁선수 슈팅동작의 일관성을 판단하는 모델** 생성    
  
  
**1차년도 :** 선수들의 **슈팅동작을 모델이 판단**하여 **어떤 선수의 슈팅**인지 예측하는 모델    
**2차년도 :** **선수 개인의 슈팅동작** 중 **일관성 정도(안정 동작)를 예측**하는 모델    
    
    
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

\* **형태변환된 json 자료를 다시 불러들여 작업** 하는 이유는 각 슈팅동작(`1개의 json 파일`)에서 **프레임별로 정규화**가 이루어져야 하기 때문(`영상 별로 카메라와 선수들의 거리, 프레임 내에서의 선수위치가 다름`)  
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
  - batch_size = 1
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
