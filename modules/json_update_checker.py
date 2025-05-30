import json
import os
from pathlib import Path

DATA_DIR = Path("./data/json")
RECORD_PATH = Path("file_record.json")


# 👇 여기에 처리할 작업 함수 정의
def process_new_json(folder_id, json_file_path):
    print(f"[{folder_id}] 처리 중: {json_file_path}")
    # 실제 처리할 코드 삽입
    # 예: json 파일 읽고 전처리 또는 분석 작업 수행
    pass


def load_previous_record(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {}


def save_current_record(record, path):
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"📁 파일 리스트 저장 완료: {path}")


def scan_current_json_files(data_dir):
    """
    data/json/ 하위의 각 사람 폴더별로 JSON 파일 목록을 딕셔너리로 리턴
    """
    all_folders = {}
    for folder in data_dir.iterdir():
        if folder.is_dir():
            json_files = sorted([f.name for f in folder.glob("*.json") if f.is_file()])
            all_folders[folder.name] = json_files
    return all_folders


def check_and_process_new_files():
    prev_record = load_previous_record(RECORD_PATH)
    current_record = scan_current_json_files(DATA_DIR)

    for folder_id, current_files in current_record.items():
        prev_files = prev_record.get(folder_id, [])

        # 새 파일 판별
        new_files = set(current_files) - set(prev_files)

        if not new_files:
            print(f"[{folder_id}] ✅ 변화 없음")
            continue

        print(f"[{folder_id}] 🆕 새로운 파일 발견: {len(new_files)}개")
        for file_name in sorted(new_files):
            file_path = DATA_DIR / folder_id / file_name
            process_new_json(folder_id, file_path)

    # 모든 결과 저장
    save_current_record(current_record, RECORD_PATH)


if __name__ == "__main__":
    check_and_process_new_files()
