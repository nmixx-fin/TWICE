import os
import argparse

def remove_files_by_name(target_filename: str, directory: str = "."):
    removed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file == target_filename:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"✅ Deleted: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
    if removed_count == 0:
        print("No matching files found.")
    else:
        print(f"총 {removed_count}개의 파일이 삭제되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="특정 이름의 파일을 디렉토리 내에서 모두 삭제합니다.")
    parser.add_argument("filename", help="삭제할 파일 이름 (예: target.txt)")
    parser.add_argument("--dir", default=".", help="탐색할 디렉토리 (기본값: 현재 디렉토리)")

    args = parser.parse_args()
    remove_files_by_name(args.filename, args.dir)
