#!/bin/bash

FILE_NAME="DATA.zip"

# gdown 설치 함수
install_gdown() {
  echo "gdown 모듈이 설치되어 있지 않습니다. 설치를 진행합니다..."
  pip install gdown
  if [ $? -ne 0 ]; then
    echo "gdown 모듈 설치 실패. 스크립트를 종료합니다."
    exit 1
  fi
}

# gdown 설치 확인 및 설치
if ! python3 -c "import gdown" >/dev/null 2>&1; then
  install_gdown
fi

# gdown을 사용하여 파일 다운로드
echo "Google Drive에서 파일을 다운로드하고 있습니다..."
python3 ./modules/download_data.py



# 압축 해제
echo "${FILE_NAME}의 압축을 해제하고 있습니다..."
if command -v unzip > /dev/null 2>&1; then
  unzip -o ${FILE_NAME} -d extracted_files
  if [ $? -eq 0 ]; then
    echo "압축 해제 완료. 파일들은 'extracted_files' 디렉토리에 위치합니다."
  else
    echo "압축 해제 실패. 압축 해제 도중 문제가 발생했습니다."
    exit 1
  fi
else
  echo "unzip이 설치되어 있지 않습니다. 압축 해제를 위해 unzip을 설치합니다."
  sudo apt-get update
  sudo apt-get install unzip -y
  if [ $? -ne 0 ]; then
    echo "unzip 설치 실패. 스크립트를 종료합니다."
    exit 1
  fi
  unzip -o ${FILE_NAME} -d DATA
  if [ $? -eq 0 ]; then
    echo "압축 해제 완료. README 의 메뉴얼대로 다음 과정을 수행하세요."
  else
    echo "압축 해제 실패. 압축 해제 도중 문제가 발생했습니다."
    exit 1
  fi
fi

