import os
import random

dataset_dir = "/home/dong/projects/IntegratedPIFu/data_list/normal/dataset.txt"
val_output = "/home/dong/projects/IntegratedPIFu/data_list/normal/val.txt"
test_output = "/home/dong/projects/IntegratedPIFu/data_list/normal/test.txt"
train_output = "/home/dong/projects/IntegratedPIFu/data_list/normal/train.txt"

def split_dataset(data_file, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    데이터셋을 훈련, 검증, 테스트 세트로 분할하는 함수

    Args:
        data_file: 데이터셋 파일 경로
        train_ratio: 훈련 데이터 비율 (기본값: 0.6)
        val_ratio: 검증 데이터 비율 (기본값: 0.2)
        test_ratio: 테스트 데이터 비율 (기본값: 0.2)
    """
    try:
        # 텍스트 파일에서 모든 라인 읽기
        with open(data_file, 'r') as f:
            all_samples = [line.strip() for line in f.readlines()]
            
        if not all_samples:
            print("The file is empty.")
            return
        
        # 전체 샘플 수
        total_samples = len(all_samples)
        
        # 각 세트의 크기 계산
        num_test = int(total_samples * test_ratio)
        num_val = int(total_samples * val_ratio)
        num_train = total_samples - (num_test + num_val)  # 나머지는 훈련셋으로
        
        # 전체 데이터를 무작위로 섞기
        random.shuffle(all_samples)
        
        # 데이터 분할
        test_samples = all_samples[:num_test]
        val_samples = all_samples[num_test:num_test + num_val]
        train_samples = all_samples[num_test + num_val:]
        
        # 파일에 쓰기 함수
        def write_to_file(samples, filepath):
            with open(filepath, 'w') as f:
                for sample in samples:
                    f.write(f"{sample}\n")
        
        # 각 세트를 파일로 저장
        write_to_file(train_samples, train_output)
        write_to_file(val_samples, val_output)
        write_to_file(test_samples, test_output)
        
        print(f"Dataset split complete:")
        print(f"Training samples: {len(train_samples)} ({len(train_samples)/total_samples:.1%})")
        print(f"Validation samples: {len(val_samples)} ({len(val_samples)/total_samples:.1%})")
        print(f"Test samples: {len(test_samples)} ({len(test_samples)/total_samples:.1%})")
        
    except FileNotFoundError:
        print(f"The file {data_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 함수 실행
split_dataset(dataset_dir)
