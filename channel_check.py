import OpenEXR
import Imath

"""
노멀 맵 채널 수 오류가 발생하여, 노멀 맵의 채널 수와 해상도를 확인하기 위해 만든 스크립트
"""

def get_exr_info(file_path):
    # EXR 파일 열기
    exr_file = OpenEXR.InputFile(file_path)
    
    # 채널 정보 가져오기
    channel_names = exr_file.header()['channels'].keys()
    channel_count = len(channel_names)
    
    # 해상도 가져오기
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    
    print(f"채널 개수: {channel_count}")
    print(f"채널 목록: {list(channel_names)}")
    print(f"해상도: {width} x {height}")
    
    return channel_count, (width, height)

# 예시 사용
file_path = "/home/public/IntegratedPIFu/rendering_script/buffer_normal_maps_of_full_mesh/0000/rendered_nmlB_000.exr"
get_exr_info(file_path)