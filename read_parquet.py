import pandas as pd
import os
import sys

def list_files(path='.'):
    print(f"Files in {os.path.abspath(path)}:")
    for f in os.listdir(path):
        print(f"  {f}")

def read_parquet(file_path):
    try:
        print(f"Trying to read: {file_path}")
        df = pd.read_parquet(file_path)
        print("\nDataFrame Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumns:")
        print(df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")
        return None

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    list_files()
    
    # 현재 디렉토리의 모든 parquet 파일 시도
    parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
    
    if parquet_files:
        for file in parquet_files:
            print(f"\nProcessing file: {file}")
            read_parquet(file)
    else:
        print("No parquet files found in current directory")
        
        # 특정 경로 시도
        specific_paths = [
            'train-00000-of-00001.parquet',
            'train-00000-of-00001 (1).parquet',
            'train-00000-of-00001 (2).parquet',
            'c:/Users/hwyew/Downloads/Financial_Embedding/TWICE/train-00000-of-00001.parquet'
        ]
        
        for path in specific_paths:
            if os.path.exists(path):
                print(f"\nFound file at: {path}")
                read_parquet(path) 