

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


**프로세스 구조에 명시적 경로**  
Data_path = os.path.join(os.getcwd(), "Data")  
Record_path = os.path.join(Data_path, "record.json")  
Json_path = os.path.join(Data_path, "Jsons")  
Video_path = os.path.join(Data_path, "Videos")  
Npy_path = os.path.join(Data_path, "npy")  
Vis_path = os.path.join(Data_path, "visual_temp")  
  
  
---  
국가대표 양궁선수들의 슈팅영상에서 HPE(Human Pose Estimation)를 이용하여 관절포인트를 추출하고 동작의 일관성을 판단하는 모델 생성    
구체적으로 **(1)관절포인트를 추출한 Json 파일을 학습모델 자료로 생성 및 로드**하는 모듈과   
**(2)생성된 학습 자료세트를 활용하여 양궁선수 슈팅동작의 일관성을 판단하는 모델**  
**(3)모델의 예측값과 동작일관성의 요소별 가중치 차이를 시각화하는 프로세스** 모듈
  
  
  
**프로세스**  (2025-06-04일 기준)  

- data_collect.py :  
```
> python data_collect.py
Data/Jsons/ 신규 json 파일 확인 (새로운 선수 & 기존 선수 새로운 슈팅 모두 포함)  
새로운 파일 탐색 시 json 파일 전처리  
선수별 슈팅자료 누적(npy 파일 생성 or 업데이트)  
작업 json 리스트 업 (예외처리 파일(900프레임 이상, 관절포인트 유실 등) 중복 탐지 방지)
```
  
- model_learning.py:  
```
> python model_learning.py {CODE}  
> python model_learning.py 48 (48번 선수 자료를 학습할 경우)  
선수별 슈팅 누적자료(npy 파일)를 학습하는 코드  
신규학습과 기존 모델 가중치 업데이트는 하이퍼파라미터 다름  
최소 1000건 필요로 되어 있음 (테스트시 이를 0으로 변경)  
  
model_learning.py 66번줄~  
# 학습파라미터 설정 및 학습실행  
if y_train.size > 1000:      # 이 부분을 if y_train.size > 0: 으로 수정  
    # Train the BiGRU model  
    model, history, test_loss, test_acc = modules.train_or_finetune_archery_model(  
        player_code=player_code,  
        x_train=x_train,  
        y_train=y_train,  
        x_test=x_test,  
        y_test=y_test,  
    )  
```
  
- visualization.py:  
```
> python visualization.py {file}  
> python visualization.py 46_999_091122_01.mp4   (.mp4 or .json)  
46_999_091122_01 영상자료와 json 파일, x_train_46.npy, 46_model.keras 를 사용하여 시각화영상 생성  
원본 영상, 원본 json, 누적 npy, 학습모델 모두 필요  
Data/visual_temp 경로에 시각화 영상 생성  
```


