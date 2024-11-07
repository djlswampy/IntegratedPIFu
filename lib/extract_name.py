import os
import argparse  


directory_path = '/home/public/data/integratedpifu_data/buffer_normal_maps_of_full_mesh'
output_file='/home/dong/projects/IntegratedPIFu/data_list/normal/dataset.txt'


def extract_directory_names(directory_path, limit, output_file):
    """
    디렉토리에서 지정된 개수만큼의 디렉토리명을 추출하여 파일로 저장하는 함수
    
    Args:
        directory_path (str): 검색할 디렉토리 경로
        limit (int, optional): 추출할 디렉토리 개수 (None인 경우 모든 디렉토리 추출)
        output_file (str): 결과를 저장할 파일명
    """
    try:
        # 디렉토리 존재 확인
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # 디렉토리 이름만 추출
        dir_names = [name for name in os.listdir(directory_path) 
                    if os.path.isdir(os.path.join(directory_path, name))]
        
        # 전체 디렉토리 수 출력
        total_dirs = len(dir_names)
        print(f"총 발견된 디렉토리 수: {total_dirs}")

        # 제한 개수 적용
        if limit is not None:
            if limit <= 0:
                raise ValueError("Limit must be a positive number")
            dir_names = dir_names[:limit]
            print(f"요청한 개수만큼 추출: {limit}")

        # 추출된 디렉토리 이름 출력
        print("\n추출된 디렉토리 목록:")
        for dir_name in dir_names:
            print(dir_name)

        # 파일로 저장
        with open(output_file, 'w') as f:
            for dir_name in dir_names:
                f.write(dir_name + '\n')
        
        print(f"\n총 {len(dir_names)}개의 디렉토리 이름이 {output_file}에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")



def main():
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='디렉토리에서 지정된 개수만큼의 디렉토리명을 추출하는 스크립트')
    
    # 인자 추가
    parser.add_argument('-i', '--input', 
                    help='검색할 디렉토리 경로',
                    default=directory_path)
    
    parser.add_argument('-l', '--limit', 
                       type=int, 
                       help='추출할 디렉토리 개수 (기본값: 모든 디렉토리)',
                       default=None)
    
    parser.add_argument('-o', '--output', 
                       help='결과를 저장할 파일명 (기본값: directory_names.txt)',
                       default=output_file)

    # 인자 파싱
    args = parser.parse_args()

    # 함수 실행
    extract_directory_names(args.input, args.limit, args.output)

if __name__ == "__main__":
    main()