import json
import os
from pathlib import Path

parent_path = Path.cwd().parent

DATA_DIR = Path(parent_path, "Data/Jsons/")
RECORD_PATH = Path(parent_path, "Data/record.json")


def load_previous_record(path):
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            return set(json.load(f))
    return set()


def save_current_record(record_set, path):
    with open(path, "w") as f:
        json.dump(sorted(list(record_set)), f, indent=2)
    print(f"📄 파일 리스트 저장 완료: {path}")


def scan_all_json_files(data_dir):
    """
    data/json/ 하위 모든 json 파일을 "48/48_01.json" 형식으로 리스트업
    """
    file_list = []
    for folder in data_dir.iterdir():
        if folder.is_dir():
            for json_file in folder.glob("*.json"):
                rel_path = os.path.join(folder.name, json_file.name)
                file_list.append(rel_path)
    return sorted(file_list)
