import OpenEXR
import Imath
import numpy as np
import imageio
import os
import glob

"""
디렉토리 내 모든 EXR 형식의 노멀 맵을 PNG 이미지로 변환하여 저장하는 스크립트
"""

def exr_to_png(input_path, output_path):
    # EXR 파일 열기
    exr_file = OpenEXR.InputFile(input_path)
    
    # EXR 헤더에서 채널 정보 가져오기
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 각 채널 읽기
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    x = np.frombuffer(exr_file.channel('X', FLOAT), dtype=np.float32)
    y = np.frombuffer(exr_file.channel('Y', FLOAT), dtype=np.float32)
    z = np.frombuffer(exr_file.channel('Z', FLOAT), dtype=np.float32)
    
    # 채널 데이터를 합쳐 이미지 배열 생성
    xyz = np.zeros((height, width, 3), dtype=np.float32)
    xyz[..., 0] = np.reshape(x, (height, width))
    xyz[..., 1] = np.reshape(y, (height, width))
    xyz[..., 2] = np.reshape(z, (height, width))
    
    # (-1, 1) 범위를 (0, 1) 범위로 매핑
    xyz = (xyz + 1) / 2
    
    # 0~255 범위로 변환하여 저장
    xyz_uint8 = (xyz * 255).astype(np.uint8)
    
    imageio.imwrite(output_path, xyz_uint8)
    print(f"변환 완료: {output_path}")

def convert_all_exr_in_directory(subject_id, base_input_dir, base_output_dir):
    input_dir = os.path.join(base_input_dir, subject_id)
    output_dir = os.path.join(base_output_dir, subject_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 디렉토리에서 모든 .exr 파일 찾기
    exr_files = glob.glob(os.path.join(input_dir, "*.exr"))
    
    for exr_file in exr_files:
        # 각 EXR 파일에 대해 출력 파일 경로 설정
        input_filename = os.path.basename(exr_file)
        output_filename = os.path.splitext(input_filename)[0] + "_normal.png"
        output_path = os.path.join(output_dir, output_filename)
        
        exr_to_png(exr_file, output_path)

# 사용 예제
subject_id = '0000'  # 변환할 서브젝트 ID
base_input_dir = '/home/public/IntegratedPIFu/rendering_script/buffer_normal_maps_of_full_mesh'  # EXR 파일이 있는 기본 입력 디렉토리
base_output_dir = '/home/public/IntegratedPIFu/get_normal_image'  # PNG 파일을 저장할 기본 출력 디렉토리

convert_all_exr_in_directory(subject_id, base_input_dir, base_output_dir)